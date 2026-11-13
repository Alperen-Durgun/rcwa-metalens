"""Command-line interface.

    rcwa-metalens list                         # experiment types
    rcwa-metalens help <type>                  # parameters of one type, with defaults
    rcwa-metalens <type> [key=value ...]       # run one experiment
    rcwa-metalens run --all | <note.md>        # note workflow (see rcwa/experiments.py)

Options:  --out DIR   write experiments/reports/outputs under DIR (same as $RCWA_PROJECT_DIR)
          --name N    name the figure/report files N.png / N.md (default CLI_<type>, no .md)
          --version

From a clone without installing:  python metadesign.py <same arguments>
"""
import os, sys, re, inspect, difflib


def utf8_stdio():
    """Make stdout/stderr UTF-8. Reports contain λ, Σ, θ, ✅ ...; on Windows a redirected or
    legacy console defaults to cp1252 and printing them would raise UnicodeEncodeError."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _balanced(src, start):
    """Text of the argument that starts at src[start], up to the matching ',' or ')'."""
    depth = 0; i = start; quote = None
    while i < len(src):
        c = src[i]
        if quote:
            if c == quote:
                quote = None
        elif c in "\"'":
            quote = c
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            if depth == 0:
                break
            depth -= 1
        elif c == "," and depth == 0:
            break
        i += 1
    return src[start:i].strip()


def experiment_params(name):
    """Parameters an experiment type reads, with their default expressions, in source order.
    Returns a list of (key, default_text). Derived from the runner's source, so it never
    goes out of date."""
    from . import experiments as R
    fn = R.RUNNERS[name]; src = inspect.getsource(fn)
    for helper in re.findall(r"\b(_build_lens)\(fm\)", src):
        src += "\n" + inspect.getsource(getattr(R, helper))
    out = {}
    for m in re.finditer(r'(?:num\(fm,\s*|fm\.get\()"(\w+)"\s*(,\s*)?', src):
        key = m.group(1)
        default = _balanced(src, m.end()) if m.group(2) else "(required)"
        out.setdefault(key, default)
    return list(out.items())


def _doc_line(fn):
    doc = (fn.__doc__ or "").strip().splitlines()
    return doc[0].strip() if doc else ""


def _print_types(R):
    print(f"Experiment types ({len(R.RUNNERS)}):")
    for n, fn in R.RUNNERS.items():
        print(f"  {n:28s} {_doc_line(fn)}")
    print("\nrcwa-metalens help <type>   shows the parameters of a type.")


def _print_help(R, name):
    fn = R.RUNNERS[name]
    print(f"{name} — {_doc_line(fn) or 'experiment'}\n")
    print("Parameters (key=value; lengths in um, angles in degrees):")
    for k, d in experiment_params(name):
        print(f"  {k:14s} default: {d}")
    print(f"\nExample:  rcwa-metalens {name} " + " ".join(f"{k}=..." for k, _ in experiment_params(name)[:2]))


def main(argv=None):
    utf8_stdio()
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--version" in argv:
        from . import __version__
        print(__version__); return 0
    if "--out" in argv:
        i = argv.index("--out")
        if i + 1 >= len(argv):
            print("--out needs a directory"); return 2
        os.environ["RCWA_PROJECT_DIR"] = argv[i + 1]; del argv[i:i + 2]
    name = None
    if "--name" in argv:                       # output file name (figure, report) instead of CLI_<type>
        i = argv.index("--name")
        if i + 1 >= len(argv):
            print("--name needs a value"); return 2
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", argv[i + 1]); del argv[i:i + 2]
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__); return 0
    from . import experiments as R
    names = list(R.RUNNERS)
    cmd = argv[0]
    if cmd in ("list", "--list"):
        _print_types(R); return 0
    if cmd == "run":
        R.main(["run_experiment"] + (argv[1:] or ["--all"])); return 0
    if cmd == "--menu":
        for i, n in enumerate(names):
            print(f"[{i:2d}] {n}")
        try:
            cmd = names[int(input("Select (number): ").strip())]
        except Exception:
            print("Invalid selection."); return 1
        raw = input("Parameters (key=val separated by spaces, empty=defaults): ").strip()
        argv = [cmd] + raw.split()
    if cmd == "help":
        if len(argv) < 2:
            print(__doc__); return 0
        cmd = argv[1]; argv = [cmd, "--help"]
    if cmd not in R.RUNNERS:
        near = difflib.get_close_matches(cmd, names, n=3)
        print(f"Unknown experiment type: {cmd}" + (f"  (did you mean: {', '.join(near)}?)" if near else ""))
        print("Run 'rcwa-metalens list' to see all types."); return 1
    if any(a in ("-h", "--help") for a in argv[1:]):
        _print_help(R, cmd); return 0
    known = [k for k, _ in experiment_params(cmd)]
    fm = {}
    for a in argv[1:]:
        if "=" not in a:
            print(f"Cannot parse '{a}' — parameters are given as key=value."); return 2
        k, v = a.split("=", 1)
        if k not in known:
            near = difflib.get_close_matches(k, known, n=2)
            print(f"Unknown parameter '{k}' for {cmd}" + (f" (did you mean {', '.join(near)}?)" if near else "")
                  + f". Run 'rcwa-metalens help {cmd}'."); return 2
        fm[k] = v
    print(f"Running: {cmd}  parameters={fm or 'defaults'}")
    slug = name or ("CLI_" + cmd)
    fig, body, summary = R.RUNNERS[cmd](fm, slug)
    print("\n=== RESULT ===\n" + body)
    print(f"\nSummary: {summary}")
    print(f"Figure:  {fig}")
    if name:                                   # --name: also keep the report text next to the figure
        md = os.path.splitext(fig)[0] + ".md"
        with open(md, "w", encoding="utf-8") as f:
            f.write(f"# {cmd} — {name}\n\nParameters: {fm or 'defaults'}\n\n{body}\n\n**Summary:** {summary}\n")
        print(f"Report:  {md}")
    return 0


def main_entry():
    raise SystemExit(main())


if __name__ == "__main__":
    main_entry()
