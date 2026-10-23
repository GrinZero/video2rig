"""Seedance clips straight from the start frame, through BytePlus ModelArk (the official Seedance API: pay per token,
no app queue, several clips at once). The start frame goes in as the video's first frame, the prompt from a text file,
and the clips land where the tool reads them: clips/<clip>.mp4, with clips/<clip>.json beside it (task id, prompt,
tokens used).

    python -m video2rig.generate my_wolf run01 --prompt prompts/run01.txt
    python -m video2rig.generate my_wolf run01 --prompt prompts/run01.txt --variants 3   # run01-1.mp4 .. run01-3.mp4
    python -m video2rig.generate my_wolf run01 --resume     # bring home the clips of a run that stopped

The API key comes from BYTEPLUS_API_KEY or the file ~/.byteplus-key (ModelArk console -> API Key Management; the
model must be activated there first). Every clip is paid: 720p 5 s is roughly 110k tokens.
A clip that does not fit in what is left of the resource pack (~/.byteplus-pack.json) is refused: pay-per-use
stays off unless that file says otherwise.

The task ids are written to clips/<clip>.task.json the moment BytePlus takes the tasks: a paid task goes on at
BytePlus whatever happens here. While waiting, network errors, timeouts and HTTP 429 / 5xx are tried again with a
growing pause (15 s doubling up to 5 min, 8 times in a row at most). If the run still stops (or the computer
sleeps), --resume picks up the saved task ids, waits, downloads and updates the pack ledger without paying again.
The task file is deleted once every clip in it is home. A task id you only have from a printout goes in by hand:
clips/run01.task.json = {"tasks": {"run01": "<task id>"}}, then --resume.
"""
import argparse
import base64
import http.client
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from .common import Project

API = 'https://ark.ap-southeast.bytepluses.com/api/v3/contents/generations/tasks'
MODEL = 'dreamina-seedance-2-5-260628'


def key():
    k = os.environ.get('BYTEPLUS_API_KEY')
    if not k:
        f = Path.home() / '.byteplus-key'
        if not f.exists():
            raise SystemExit('no API key: set BYTEPLUS_API_KEY or write it to ~/.byteplus-key')
        k = f.read_text(encoding='utf-8')
    return k.strip()


# --- the BytePlus resource pack guard: never pay per use without saying so ---------------------------------------
# The pack is charged by value: one API token costs ~1.67 pack tokens (2026-10-09, $0.0107/K API against $0.0064/K
# pack). ~/.byteplus-pack.json holds what is left in the pack (update it from the ModelArk console or its e-mails);
# a clip that does not fit in it is refused, unless "allow_pay_per_use" is true there.
LEDGER = Path.home() / '.byteplus-pack.json'
API_TOKENS_PER_S = {'480p': 9700, '720p': 21700, '1080p': 48800}     # measured: 720p 10 s = 216 900 tokens


def pack_read():
    if not LEDGER.exists():
        raise SystemExit(f'no {LEDGER}: write what is left in the BytePlus pack there first, e.g. '
                         '{"remaining_pack_ktokens": 674.0, "pack_per_api_token": 1.67, "allow_pay_per_use": false}')
    return json.loads(LEDGER.read_text(encoding='utf-8'))


def pack_check(a, n):
    led = pack_read()
    need = API_TOKENS_PER_S.get(a.resolution, 48800) * a.duration * n * float(led.get('pack_per_api_token', 1.67)) / 1000
    left = float(led['remaining_pack_ktokens'])
    print(f'pack: {left:.0f}k pack tokens left, these {n} clip(s) need about {need:.0f}k', flush=True)
    if need > left * 0.97 and not led.get('allow_pay_per_use', False):
        raise SystemExit(f'REFUSED: the BytePlus pack would run out ({left:.0f}k left, about {need:.0f}k needed) and '
                         'pay-per-use is off. Use Dreamina, a shorter clip, or a lower resolution.')


def pack_spend(usage):
    try:
        led = pack_read()
        used = (usage or {}).get('total_tokens', 0) * float(led.get('pack_per_api_token', 1.67)) / 1000
        led['remaining_pack_ktokens'] = round(float(led['remaining_pack_ktokens']) - used, 3)
        led['updated'] = time.strftime('%Y-%m-%d %H:%M')
        LEDGER.write_text(json.dumps(led, indent=1), encoding='utf-8')
        print(f'pack: {used:.0f}k pack tokens spent, {led["remaining_pack_ktokens"]:.0f}k left', flush=True)
    except Exception as e:                              # the clip is already paid for: never lose it over the ledger
        print(f'pack ledger not updated: {e}', flush=True)


POLL_S = 15                     # between two looks at the tasks
BACKOFF_S, BACKOFF_MAX_S = 15, 300
MAX_FAILS = 8                   # errors in a row on one task before the run stops (about 23 minutes of pauses)


class Transient(Exception):
    """Worth trying again: the network, a timeout, HTTP 429 or 5xx."""


class ApiError(Exception):
    """The API said no (HTTP 4xx other than 429): trying again would get the same answer."""


def call(method, url, body=None):
    data = json.dumps(body).encode('utf-8') if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        'Authorization': f'Bearer {key()}', 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        msg = f'{method} {url}: HTTP {e.code} {e.read().decode("utf-8", "replace")[:600]}'
        raise (Transient if e.code == 429 or e.code >= 500 else ApiError)(msg) from e
    except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError, http.client.HTTPException,
            json.JSONDecodeError) as e:
        raise Transient(f'{method} {url}: {type(e).__name__} {e}') from e


def download(url, path):
    """The clip to a .part file first, renamed when complete: a broken download never looks like a clip."""
    part = path.with_name(path.name + '.part')
    try:
        urllib.request.urlretrieve(url, part)
    except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError, http.client.HTTPException) as e:
        part.unlink(missing_ok=True)
        raise Transient(f'download {path.name}: {type(e).__name__} {e}') from e
    part.replace(path)


def job_path(clips, clip):
    return clips / f'{clip}.task.json'


def job_write(path, job):
    """The paid tasks not yet home; the file goes once there are none."""
    if not job['tasks']:
        path.unlink(missing_ok=True)
        return
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_bytes(json.dumps(job, indent=1).encode('utf-8'))
    tmp.replace(path)


NO_EFFECTS = ('No visual effects of any kind: no dust, no smoke, no debris, no sparks, no particles, no speed lines, '
              'no motion trails, no light trails, no motion blur, no camera shake. The blade and every limb stay sharp, '
              'the flat floor stays perfectly clean, and every foot, hand and held object stays clearly visible in '
              'every frame.')


def with_no_effects(prompt):
    """Every clip without effects: Seedance adds dust clouds at every hoof strike and sparks and trails to every
    sword cut, and the reader, taught on clean renders, guesses what they hide with full confidence (a centaur's
    hooves under dust, 2026-10-19). The sentence goes at the end of every prompt unless it is there already."""
    return prompt if 'no visual effects' in prompt.lower() else prompt.rstrip() + ' ' + NO_EFFECTS


def slow_motion(time_scale):
    """The sentence that asks for the clip in slow motion: `time_scale` of the real speed (.5: half speed)."""
    speed = 'half' if abs(time_scale - .5) < 1e-6 else f'{time_scale:g} times'
    return (f'The whole action is filmed in slow motion at {speed} its real speed, at one steady pace from the first '
            f'frame to the last.')


def prompt_as_sent(prompt, keep_effects=False, time_scale=1.0):
    """The prompt as it goes to Seedance (and into clips/<clip>.json): a slow-motion clip (time_scale under 1) gets
    the slow-motion sentence, every clip the no-effects one after it. A strike in slow motion spreads over more
    frames, so the reader sees a fast hand in more places and less smeared; the motion is written at the clip's
    times x time_scale (creature.json, clips.<clip>.time_scale), so it plays at normal speed."""
    prompt = prompt.rstrip()
    if time_scale < 1 and 'slow motion' not in prompt.lower():
        prompt += ' ' + slow_motion(time_scale)
    return prompt if keep_effects else with_no_effects(prompt)


def submit(start, prompt, a):
    b64 = base64.b64encode(start.read_bytes()).decode('ascii')
    mime = 'image/png' if start.suffix.lower() == '.png' else 'image/jpeg'
    body = {'model': a.model,
            'content': [{'type': 'text', 'text': prompt},
                        {'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{b64}'}, 'role': 'first_frame'}],
            'resolution': a.resolution, 'duration': a.duration, 'ratio': 'adaptive',
            'generate_audio': True, 'watermark': False}
    return call('POST', API, body)['id']


def wait(clips, path, job, resume_hint):
    """Look at the job's tasks until each one is home (downloaded, its .json written, the pack ledger updated) or
    ended at BytePlus; the task file shrinks with every task that is done."""
    fails = {n: 0 for n in job['tasks']}
    t0 = time.time()
    while job['tasks']:
        time.sleep(POLL_S)
        for n, t in list(job['tasks'].items()):
            try:
                r = call('GET', f'{API}/{t}')
                st = r.get('status')
                if st == 'succeeded':
                    download(r['content']['video_url'], clips / f'{n}.mp4')
            except Transient as e:
                fails[n] += 1
                if fails[n] >= MAX_FAILS:
                    raise SystemExit(f'{n}: {MAX_FAILS} errors in a row, the last: {e}\n{resume_hint}')
                pause = min(BACKOFF_S * 2 ** (fails[n] - 1), BACKOFF_MAX_S)
                print(f'{n}: {e}; trying again in {pause} s ({fails[n]} of {MAX_FAILS - 1})', flush=True)
                time.sleep(pause)
                continue
            except ApiError as e:
                raise SystemExit(f'{n}: {e}\n{resume_hint}')
            fails[n] = 0
            if st == 'succeeded':
                meta = {'task': t, **{k: job.get(k) for k in ('model', 'start', 'prompt', 'resolution', 'duration')},
                        'usage': r.get('usage'), 'seconds_to_ready': round(time.time() - job.get('submitted_s', t0))}
                (clips / f'{n}.json').write_bytes(json.dumps(meta, indent=1).encode('utf-8'))
                print(f'{n}: ready after {meta["seconds_to_ready"]} s, tokens {r.get("usage")}', flush=True)
                pack_spend(r.get('usage'))
            elif st in ('failed', 'cancelled', 'expired'):
                print(f'{n}: {st} {json.dumps(r.get("error"))}', flush=True)
            else:
                continue
            del job['tasks'][n]
            job_write(path, job)


def resume(clips, clip):
    path = job_path(clips, clip)
    if not path.exists():
        raise SystemExit(f'{path} not found: no unfinished tasks for {clip}')
    job = json.loads(path.read_text(encoding='utf-8'))
    for n in list(job['tasks']):
        if (clips / f'{n}.mp4').exists():                # home already; spending it again would be wrong
            print(f'{n}: {clips / n}.mp4 is already here, skipped (if the run stopped right after the download, '
                  f'check that {LEDGER} took its tokens off)', flush=True)
            del job['tasks'][n]
    job_write(path, job)
    print(f'resuming {len(job["tasks"])}: ' + ', '.join(f'{n} {t}' for n, t in job['tasks'].items()), flush=True)
    return path, job


def main():
    ap = argparse.ArgumentParser(prog='python -m video2rig.generate')
    ap.add_argument('project')
    ap.add_argument('clip')
    ap.add_argument('--prompt', default=None, help='a text file with the prompt')
    ap.add_argument('--keep-effects', action='store_true', help='do not add the no-effects sentence to the prompt '
                    '(dust, sparks, trails hide what the reader has to see)')
    ap.add_argument('--start', default=None, help='the first frame (default: start/start.png of the project)')
    ap.add_argument('--slow-motion', type=float, default=1.0, metavar='SCALE', help='ask for the clip in slow motion '
                    "at SCALE of the real speed (0.5: half speed); give the clip the same time_scale in creature.json")
    ap.add_argument('--dry-run', action='store_true', help='print the prompt as it would be sent and stop (a clip '
                    'made in another app, such as Dreamina)')
    ap.add_argument('--variants', type=int, default=1)
    ap.add_argument('--resolution', default='720p')
    ap.add_argument('--duration', type=int, default=5)
    ap.add_argument('--model', default=MODEL)
    ap.add_argument('--resume', action='store_true', help='no new task: wait for the ones saved in '
                    'clips/<clip>.task.json by a run that stopped, download them, update the pack ledger')
    a = ap.parse_args()
    pr = Project(a.project)
    clips = pr.root / 'clips'
    clips.mkdir(exist_ok=True)
    resume_hint = (f'The paid task(s) go on at BytePlus. Bring them home later with:\n'
                   f'  python -m video2rig.generate {a.project} {a.clip} --resume')
    t0 = time.time()
    if a.resume:
        path, job = resume(clips, a.clip)
    else:
        if not a.prompt:
            ap.error('--prompt is required (unless --resume)')
        path = job_path(clips, a.clip)
        if path.exists():
            raise SystemExit(f'{path} holds tasks of an earlier run that are not home yet: add --resume')
        start = Path(a.start) if a.start else pr.start_dir / 'start.png'
        prompt = Path(a.prompt).read_text(encoding='utf-8').strip()
        prompt = prompt_as_sent(prompt, a.keep_effects, a.slow_motion)   # stored as sent (clips/<clip>.json)
        if a.dry_run:
            print(prompt)
            return
        names = [a.clip] if a.variants == 1 else [f'{a.clip}-{i + 1}' for i in range(a.variants)]
        for n in names:
            if (clips / f'{n}.mp4').exists():
                raise SystemExit(f'{clips / n}.mp4 exists: pick another clip name')
        pack_check(a, len(names))
        job = {'model': a.model, 'start': str(start), 'prompt': prompt, 'resolution': a.resolution,
               'duration': a.duration, 'submitted': time.strftime('%Y-%m-%d %H:%M'), 'submitted_s': round(t0),
               'tasks': {}}
        for n in names:
            try:
                job['tasks'][n] = submit(start, prompt, a)
            except (Transient, ApiError) as e:               # never sent twice: a lost answer may hide a paid task
                note = ('\nThe task may have been created all the same: look in the ModelArk console before trying '
                        'again.') if isinstance(e, Transient) else ''
                raise SystemExit(f'submit {n}: {e}{note}' + (f'\n{resume_hint}' if job['tasks'] else ''))
            finally:
                job_write(path, job)                    # each task id saved the moment BytePlus has it
        print(f'submitted {len(job["tasks"])}: ' + ', '.join(f'{n} {t}' for n, t in job['tasks'].items()) +
              f' (task ids in {path})', flush=True)
    wait(clips, path, job, resume_hint)
    print(f'done in {round(time.time() - t0)} s', flush=True)


if __name__ == '__main__':
    sys.exit(main())
