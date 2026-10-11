import argparse
import importlib.util
import os
import re
import signal
import subprocess
import sys
import threading

from database import FaceDB
import tts


def parse_args():
    p = argparse.ArgumentParser(
        description="Wrapper: preloads the voice models, runs your face-check script, and when it "
                    "outputs true, greets the person and runs voice verification.")
    p.add_argument("--script", default="check_person.py", help="Filename of your original face-check script")
    p.add_argument("--id", type=int, required=True,
                   help="Database ID of the person (used for the face check and the voice check)")
    p.add_argument("--voice-db", default="people.db", help="Path to the SQLite database")
    p.add_argument("--verify-script", default="check_voice.py", help="Filename of the voice verification script")
    p.add_argument("--passphrase", default="open sesame", help="Passphrase for the voice check")
    p.add_argument("--threshold", type=float, default=0.45, help="Minimum voice similarity to accept")
    p.add_argument("--trigger", default=r"\btrue\b",
                   help="Regex (case-insensitive) that, when seen in the face script's output, starts the voice check")
    p.add_argument("--max-attempts", type=int, default=3,
                   help="How many voice attempts the person gets before access is denied")
    p.add_argument("--speech-phrase", default=None,
                   help="Greeting spoken before verification (default: uses the name stored for --id)")
    return p.parse_known_args()


def pause_process(proc):
    """Freeze the face script (all threads) so it uses no CPU. Linux/macOS only."""
    if not hasattr(signal, "SIGSTOP"):
        print("[speaker] Pausing the face script isn't supported on this OS; leaving it running.")
        return False
    try:
        os.kill(proc.pid, signal.SIGSTOP)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def resume_process(proc):
    try:
        os.kill(proc.pid, signal.SIGCONT)
    except (ProcessLookupError, PermissionError, AttributeError):
        pass


def stop_process(proc):
    """Shut the face script down (unfreezing it first so it can receive the signal)."""
    if proc.poll() is not None:
        return
    resume_process(proc)
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def lookup_name(db_path, pid):
    db = FaceDB(db_path)
    try:
        return db.label(pid) if pid in db.people else None
    finally:
        db.close()


def load_verify_module(path):
    """Import the voice script by file path, whatever it is called."""
    spec = importlib.util.spec_from_file_location("voice_verify", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not (hasattr(module, "load_models") and hasattr(module, "verify_voice")):
        sys.exit(f"Error: '{path}' is an old version (no load_models/verify_voice). "
                 f"Replace it with the latest check_voice.py.")
    return module


def main():
    args, unknown_args = parse_args()

    if not os.path.isfile(args.verify_script):
        sys.exit(f"Error: voice script '{args.verify_script}' not found in {os.getcwd()}. "
                 f"Use --verify-script <filename> or rename the file to check_voice.py.")

    name = lookup_name(args.voice_db, args.id)
    if name is None:
        sys.exit(f"Error: ID {args.id} not found in {args.voice_db}.")

    phrase = args.speech_phrase or f"Welcome back {name}, please speak the passphrase to proceed"
    retry_phrase = "Verification failed, please try again"
    denied_phrase = "Access denied."
    granted_phrase = "Access granted."
    verify_mod = load_verify_module(args.verify_script)

    state = {}

    def preload_models():
        try:
            state["models"] = verify_mod.load_models()
        except Exception as e:
            state["models_error"] = e

    def preload_greeting():
        state["greeting"] = tts.render(phrase)
        state["retry"] = tts.render(retry_phrase)
        state["denied"] = tts.render(denied_phrase)
        state["granted"] = tts.render(granted_phrase)

    print("[speaker] Loading voice models and greeting (face check starts after this)...")
    loaders = [threading.Thread(target=preload_models), threading.Thread(target=preload_greeting)]
    for t in loaders:
        t.start()
    for t in loaders:
        t.join()

    if "models_error" in state:
        sys.exit(f"[speaker] Could not load voice models: {state['models_error']}")
    print("[speaker] Everything loaded. Starting face check.")

    # The face script gets the same ID
    cmd = [sys.executable, "-u", args.script, "--id", str(args.id)] + unknown_args
    print(f"[speaker] Launching: {' '.join(cmd)}")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0  # raw bytes, no line buffering
        )
    except FileNotFoundError:
        sys.exit(f"Error: Could not find script '{args.script}'. Make sure it's in the same directory.")

    trigger = threading.Event()
    trigger_re = re.compile(args.trigger, re.IGNORECASE)

    def read_output():
        fd = process.stdout.fileno()
        tail = ""
        while True:
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            sys.stdout.write(text)
            sys.stdout.flush()
            if not trigger.is_set():
                tail = (tail + text)[-300:]  
                if trigger_re.search(tail):
                    trigger.set()

    threading.Thread(target=read_output, daemon=True).start()

    def say(key, text):
        audio = state.get(key)
        if audio is not None:
            tts.play(audio)
        else:
            tts.speak_blocking(text)

    exit_code = 0
    try:
        while process.poll() is None and not trigger.is_set():
            trigger.wait(0.2)

        if trigger.is_set():
            print(f"\n[speaker] Triggered! Speaking: '{phrase}'")

            paused = pause_process(process)
            if paused:
                print("[speaker] Face check paused.")

            ok = False
            try:
                say("greeting", phrase)
                encoder, whisper_model = state["models"]

                for attempt in range(1, args.max_attempts + 1):
                    ok = verify_mod.verify_voice(
                        args.id, encoder, whisper_model,
                        db_path=args.voice_db, passphrase=args.passphrase,
                        threshold=args.threshold, no_prompt=True)  # already greeted
                    print(f"[speaker] Attempt {attempt}/{args.max_attempts}: "
                          f"{'PASSED' if ok else 'FAILED'}")
                    if ok:
                        say("granted", granted_phrase)
                        break
                    if attempt < args.max_attempts:
                        say("retry", retry_phrase)  
                    else:
                        say("denied", denied_phrase)
            finally:
                if paused:
                    resume_process(process)

            exit_code = 0 if ok else 1
            print("[speaker] Access granted. Stopping everything." if ok
                  else "[speaker] Access denied. Stopping everything.")
        else:
            process.wait()  

    except KeyboardInterrupt:
        print("\n[speaker] Stopping wrapper...")
        exit_code = 130
    finally:
        stop_process(process)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()