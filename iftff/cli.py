"""Console entry point for the iftff package: `iftff <command> [options]`.

Installed by pip (see [project.scripts] in pyproject.toml) so the three main
actions are available from the terminal:

  iftff train      train a force field from a set of PDBs (produces an XML)
  iftff md         run molecular dynamics with a custom force field
  iftff minimize   energy-minimize a structure with a custom force field
"""
import sys

_USAGE = """iftff - implicit-function-theorem force-field optimization

usage: iftff <command> [options]

commands:
  train      train a force field from a set of PDBs  ->  trained XML
  md         run molecular dynamics with a (custom) force-field XML
  minimize   energy-minimize a structure with a (custom) force-field XML
  gui        open the graphical interface (all of the above, in a window)

Run 'iftff <command> --help' for the options of each command.
"""


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ('-h', '--help'):
        print(_USAGE)
        return 0 if argv else 1
    cmd, rest = argv[0], argv[1:]
    # hand the remaining args to the sub-command's own argparse
    sys.argv = ['iftff ' + cmd] + rest
    if cmd == 'train':
        from .train import main as run
        return run()
    if cmd == 'md':
        from .run import main as run
        return run()
    if cmd == 'minimize':
        from .run import main_minimize as run
        return run()
    if cmd == 'gui':
        from .gui import main as run
        return run()
    sys.stderr.write(f"iftff: unknown command '{cmd}'\n\n")
    print(_USAGE)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
