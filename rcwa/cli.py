"""Command-line interface.

    rcwa-metalens list                         # experiment types
    rcwa-metalens help <type>                  # parameters of one type, with defaults
    rcwa-metalens <type> [key=value ...]       # run one experiment
    rcwa-metalens run --all | <note.md>        # note workflow (see rcwa/experiments.py)
    rcwa-metalens recipe <spec.yaml> [--out DIR] [--jobs N]
                                               # specification -> recipe folder (report, GDS,
                                               # manifest, trust level); docs/recipe_pipeline.md

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
    seen = set()                               # helpers that read fm, followed recursively
    while True:
        new = set(re.findall(r"\b(_build_lens|_substrate_index)\(fm\b", src)) - seen
        if not new:
            break
        for helper in sorted(new):
            src += "\n" + inspect.getsource(getattr(R, helper)); seen.add(helper)
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
    if argv and argv[0] == "recipe":
        return recipe_main(argv[1:])
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


def recipe_main(args):
    """rcwa-metalens recipe <spec.yaml> [--out DIR] [--jobs N]. Exit code 0 = recipe written
    (status ok), 3 = specification infeasible (report written), 2 = bad arguments/spec."""
    from .recipe_run import run_recipe, SpecError, load_spec
    out = None; jobs = None; rest = []
    i = 0
    while i < len(args):
        if args[i] in ("--out", "--jobs"):
            if i + 1 >= len(args):
                print(f"{args[i]} needs a value"); return 2
            if args[i] == "--out":
                out = args[i + 1]
            else:
                try:
                    jobs = int(args[i + 1])
                except ValueError:
                    print("--jobs needs an integer"); return 2
            i += 2
        else:
            rest.append(args[i]); i += 1
    if len(rest) != 1 or rest[0] in ("-h", "--help"):
        print("usage: rcwa-metalens recipe <spec.yaml> [--out DIR] [--jobs N]\n"
              "See docs/recipe_pipeline.md and docs/recipe_spec_example.yaml."); return 0 if rest[:1] in (["-h"], ["--help"]) else 2
    try:
        spec = load_spec(rest[0])
    except (SpecError, OSError) as e:
        print(f"specification error: {e}"); return 2
    out = out or os.path.join("recipes", re.sub(r"[^A-Za-z0-9_.-]+", "_", spec["name"]))
    try:
        res = run_recipe(spec, out, n_jobs=jobs)
    except SpecError as e:
        print(f"specification error: {e}"); return 2
    print(f"\nstatus: {res['status']}" + (f"  trust level: {res['trust']['level']}" if res.get("trust") else "")
          + f"\nfolder: {out}")
    return 0 if res["status"] == "ok" else 3


def main_entry():
    raise SystemExit(main())


if __name__ == "__main__":
    main_entry()
