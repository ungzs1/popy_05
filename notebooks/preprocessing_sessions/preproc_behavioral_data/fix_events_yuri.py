"""
Fixes the behavioral event codes of monkey Yuri, so that popy.behavior_data_tools.get_behavior_yuri() can split them
into trials. The events are read from <cfg.COMBINED_DATA_PATH>/<condition>/<session>_events.npz (see the readme.md
there). The files are NOT changed: the fixes are applied to the loaded events, and every change is returned as a log
line (preproc_behavioral_data_yuri.py saves them next to the processed data). Event codes: data/code_LUV_task.txt.

Why: the codes go from the behavior computer to the recording system (NSP) as 8-bit words. Codes sent in the same ms
overlap, so a few arrive with bits missing, split in two, reordered, doubled or not at all. This happens mostly at the
trial start, where 101 (trial end), 100 (trial start) and 51/52/53 (best target) are sent together. E.g. 51/52/53 sent
while 100 is still on arrive as 19/16/17 (= 5x without the bits of 100), 100 sent while 51 is still on arrives as 68
(= 100 without the bits of 51), and 101 can arrive as 51/64/65/1 or not at all. Compared to the behavior computer's own
log (data/recordings/Yuri_behavior/behavior/<session>.csv), ~7% of the trial starts are affected (mostly only the 5x),
~0.5% have a lost or changed 100/101. The timing of the codes that arrived is exact.

Rules, applied in this order:
  Rule 1 - main task only: the events before the 250 code (start of the main task; before it there is a short other
           task with its own 100/101 codes, and in some sessions a 255) and after the 102 code (end of session) are
           removed.
  Rule 2 - trial starts are rebuilt. A trial start group is a run of events with codes in TRIAL_START_CODES, less than
           MAX_GAP_S apart, that contains a 100, or a code only seen at trial starts (START_ONLY_CODES, or 51/52/53 not
           right after a block start 8) and is followed by the trial init codes (1/2/60) within MAX_INIT_DELAY_S.
           (The 1/2 codes of the trial init burst, 1/2 codes followed by 60 within INIT_BURST_GAP_S, are never part of
           a group.) The codes of the group are replaced by [101] 100 5x:
             - 101 (end of the previous trial) only if the previous trial has no 101 yet, at the time of the group's 101
               (or of the group's first event if it is lost).
             - 100 at the time of the group's 100 (or of the group's first event if it is lost).
             - 5x = the best target of the block, i.e. the code sent with the block start (8 51/52/53, always intact in
               this data; in the behavior computer's log the 5x of every trial start equals the one of its block start).
               At the time of the 100.
  Rule 3 - lost trial start: if, after a 101, the trial init codes 1/2 come without a trial start, a 100 5x is inserted
           at the time of the first of them.
  Rule 4 - one 101 per trial: a 101 when there is no open trial (a second 101 before the next trial start) is removed.
  Rule 5 - one 125 per trial (same as rule 1 of the fix of the CSV files, data/recordings/Yuri_behavior/behavior/
           file_changes.txt): only the first 125 (target validation) of a trial is kept.
  Rule 6 - unfinished last trial: if the last trial has no 101 (session ended, or the recording stopped, during the
           trial), its events from its 100 on are removed.

Not fixed (the extraction gives NaN instead): trials where a code needed for the extraction was lost (e.g. the lever
validation 62, the target validation 125, the outcome 65/66).

Usage:
    events = load_events_yuri(session_floc)
    fixed_events, log_lines = fix_events_yuri(events)
"""
import numpy as np
import pandas as pd

TRIAL_START_CODES = {1, 4, 16, 17, 19, 36, 48, 51, 52, 53, 64, 65, 68, 96, 100, 101}  # codes seen in trial start groups
START_ONLY_CODES = {16, 17, 19, 68, 96}  # 51/52/53 or 100 with bits missing, only seen at trial starts
BEST_TARGET_CODES = (51, 52, 53)
MAX_GAP_S = 0.010  # max time between neighbor events of a trial start group
MAX_INIT_DELAY_S = 0.4  # max time from a trial start to its trial init codes (1/2 then 60) (max in the CSV logs: 0.34 s)
INIT_BURST_GAP_S = 0.003  # max time between the codes of the trial init burst (1/2 codes then 60)


def load_events_yuri(session_floc):
    """Loads the events of one session (<session>_events.npz), returns a DataFrame with columns 'time', 'sample', 'code'."""
    events = pd.DataFrame(dict(np.load(session_floc)))
    return events[['time', 'sample', 'code']]


def _event_str(events, i):
    return f"{int(events.code.iloc[i])} ({events.time.iloc[i]:.4f} s)"


def _codes_str(codes):
    return ' '.join(str(int(c)) for c in codes)


def crop_main_task(events, log):
    """Rule 1: keeps the events from the 250 code (start of the main task) to the last 102 code (end of session)."""
    code = events.code.to_numpy()
    ids_250 = np.where(code == 250)[0]
    if len(ids_250) != 1:
        raise ValueError(f"Expected one 250 code (start of the main task), found {len(ids_250)}.")
    ids_102 = np.where(code == 102)[0]
    end = ids_102[-1] + 1 if len(ids_102) > 0 else len(code)

    if ids_250[0] > 0:
        log.append(f"rule 1: removed the {ids_250[0]} events before the 250 code (start of the main task, at "
                   f"{events.time.iloc[ids_250[0]]:.4f} s), {np.sum(code[:ids_250[0]] == 100)} of them 100")
    if len(ids_102) == 0:
        log.append(f"rule 1: no 102 code (end of session): the recording ends at {events.time.iloc[-1]:.4f} s")
    elif end < len(code):
        log.append(f"rule 1: removed the {len(code) - end} events after the 102 code (end of session, at "
                   f"{events.time.iloc[ids_102[-1]]:.4f} s), codes: {sorted(set(code[end:].tolist()))}")

    return events.iloc[ids_250[0]:end].reset_index(drop=True)


def _init_burst_mask(code, time):
    """True for the 1/2 codes of the trial init bursts: 1/2 codes followed by 60, all less than INIT_BURST_GAP_S apart."""
    mask = np.zeros(len(code), dtype=bool)
    i = 0
    while i < len(code):
        if code[i] in (1, 2):
            j = i
            while j + 1 < len(code) and code[j + 1] in (1, 2) and time[j + 1] - time[j] <= INIT_BURST_GAP_S:
                j += 1
            if j + 1 < len(code) and code[j + 1] == 60 and time[j + 1] - time[j] <= INIT_BURST_GAP_S:
                mask[i:j + 1] = True
            i = j + 1
        else:
            i += 1
    return mask


def _block_best_targets(code):
    """For each event, the best target code (51/52/53) sent with the last block start (8) before it (0 if none yet)."""
    best = np.zeros(len(code), dtype=int)
    curr = 0
    for i in range(len(code)):
        if code[i] == 8 and i + 1 < len(code) and code[i + 1] in BEST_TARGET_CODES:
            curr = code[i + 1]
        best[i] = curr
    return best


def _decode_best_target(codes):
    """Best target code (51/52/53) from the codes of a trial start group, 0 if none or ambiguous."""
    decoded = {c for c in codes if c in BEST_TARGET_CODES}
    decoded |= {{19: 51, 16: 52, 17: 53, 68: 51}[c] for c in codes if c in (19, 16, 17, 68)}
    return decoded.pop() if len(decoded) == 1 else 0


def _find_trial_start_groups(code, time):
    """Rule 2: returns the (first, last) index of each trial start group (see the module docstring)."""
    is_init = _init_burst_mask(code, time)
    after_block_start = np.append(False, code[:-1] == 8) & np.isin(code, BEST_TARGET_CODES)
    in_group = np.isin(code, list(TRIAL_START_CODES)) & ~is_init & ~after_block_start

    groups = []
    i = 0
    while i < len(code):
        if not in_group[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(code) and in_group[j + 1] and time[j + 1] - time[j] < MAX_GAP_S:
            j += 1
        group = code[i:j + 1]
        if 100 in group:
            groups.append((i, j))
        elif np.isin(group, list(START_ONLY_CODES) + list(BEST_TARGET_CODES)).any():
            # without a 100, it has to be followed by the trial init codes
            if j + 1 < len(code) and code[j + 1] in (1, 2, 60) and time[j + 1] - time[j] < MAX_INIT_DELAY_S:
                groups.append((i, j))
        i = j + 1
    return groups


def rebuild_trial_starts(events, log):
    """Rules 2, 3 and 4 (see the module docstring)."""
    code, time, sample = events.code.to_numpy(), events.time.to_numpy(), events['sample'].to_numpy()
    groups = {i: j for i, j in _find_trial_start_groups(code, time)}
    block_best = _block_best_targets(code)

    rows = []  # output events, as indices of the input events (int) or new events (time, sample, code)
    trial_open = False  # a 100 was written and no 101 after it yet
    i = 0
    while i < len(code):
        if i in groups:  # rule 2
            j = groups[i]
            group = code[i:j + 1]
            best = block_best[i] if block_best[i] else _decode_best_target(group)
            new = []  # (code, index of the input event giving the time)
            if trial_open:
                new.append((101, i + int(np.argmax(group == 101)) if 101 in group else i))
            id_100 = i + int(np.argmax(group == 100)) if 100 in group else i
            new.append((100, id_100))
            if best:
                new.append((best, i + int(np.argmax(group == best)) if best in group else id_100))

            if [c for c, _ in new] == group.tolist():  # nothing to fix
                rows.extend(range(i, j + 1))
            else:
                t_100 = time[id_100]
                for c, k in new:
                    k = id_100 if c == 101 and time[k] > t_100 else k  # 101 not after the 100
                    rows.append((time[k], sample[k], c))
                note = ''
                decoded = _decode_best_target(group)
                if decoded and best and decoded != best:
                    note = f" (the group's codes suggest {decoded}, the block start says {best})"
                if not best:
                    note = " (best target unknown: no block start before and not decodable from the group)"
                log.append(f"rule 2: trial start at {t_100:.4f} s: {_codes_str(group)} -> {_codes_str([c for c, _ in new])}"
                           f"{note}  [{', '.join(_event_str(events, k) for k in range(i, j + 1))}]")
            trial_open = True
            i = j + 1
            continue

        if code[i] == 101:  # rule 4
            if trial_open:
                rows.append(i)
                trial_open = False
            else:
                log.append(f"rule 4: removed a 101 at {time[i]:.4f} s: the trial already has a 101")
            i += 1
            continue

        if code[i] in (1, 2) and not trial_open and np.any(code[:i] == 101):  # rule 3
            best = block_best[i]
            rows.append((time[i], sample[i], 100))
            if best:
                rows.append((time[i], sample[i], best))
            trial_open = True
            log.append(f"rule 3: lost trial start: inserted {_codes_str([100, best] if best else [100])} at "
                       f"{time[i]:.4f} s, before the trial init code {code[i]} that follows the 101 at "
                       f"{time[np.where(code[:i] == 101)[0][-1]]:.4f} s without a trial start")

        rows.append(i)
        i += 1

    rows = [(time[r], sample[r], code[r]) if isinstance(r, (int, np.integer)) else r for r in rows]
    return pd.DataFrame(rows, columns=['time', 'sample', 'code']).astype({'sample': sample.dtype, 'code': code.dtype})


def _trial_bounds(code):
    """(index of the 100, index of the 101 or None) of each trial."""
    bounds = []
    start = None
    for i, c in enumerate(code):
        if c == 100:
            if start is not None:
                bounds.append((start, None))
            start = i
        elif c == 101 and start is not None:
            bounds.append((start, i))
            start = None
    if start is not None:
        bounds.append((start, None))
    return bounds


def remove_repeated_125(events, log):
    """Rule 5: keeps only the first 125 (target validation) of each trial."""
    code, time = events.code.to_numpy(), events.time.to_numpy()
    drop = []
    for start, end in _trial_bounds(code):
        end = len(code) - 1 if end is None else end
        ids_125 = start + np.where(code[start:end + 1] == 125)[0]
        for k in ids_125[1:]:
            drop.append(k)
            log.append(f"rule 5: removed the 125 at {time[k]:.4f} s: second 125 in the trial starting at "
                       f"{time[start]:.4f} s (first 125 at {time[ids_125[0]]:.4f} s), trial codes: "
                       f"{_codes_str(code[start:end + 1])}")
    return events.drop(index=drop).reset_index(drop=True)


def remove_unfinished_last_trial(events, log):
    """Rule 6: removes the last trial if it has no 101."""
    code, time = events.code.to_numpy(), events.time.to_numpy()
    bounds = _trial_bounds(code)
    if not bounds or bounds[-1][1] is not None:
        return events
    start = bounds[-1][0]
    end = len(code) - 1 if code[-1] == 102 else len(code)  # keep the 102
    log.append(f"rule 6: removed the unfinished last trial (no 101), {end - start} events from {time[start]:.4f} s to "
               f"{time[end - 1]:.4f} s: {_codes_str(code[start:end])}")
    return events.drop(index=range(start, end)).reset_index(drop=True)


def fix_events_yuri(events):
    """
    Applies all rules (see the module docstring) to the events of one session (DataFrame with columns 'time',
    'sample', 'code', as loaded by load_events_yuri()). Returns the fixed events and the list of changes (strings).
    """
    log = []
    events = crop_main_task(events, log)
    events = rebuild_trial_starts(events, log)
    events = remove_repeated_125(events, log)
    events = remove_unfinished_last_trial(events, log)
    return events, log
