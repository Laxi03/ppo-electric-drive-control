"""Console and module entry point."""
import sys


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["results"]:
        from .results import main as run
        try:
            return run(args[1:])
        except (FileNotFoundError, ValueError) as error:
            from .results import parser
            parser().error(str(error))
    if args[:1] == ["thesis-figures"]:
        from .thesis_figures import main as run
        return run(args[1:])
    if args in ([], ["--help"], ["-h"]):
        from .runner import parser
        parser().print_help()
        print("\nAdditional commands: results --help, thesis-figures --help")
        return None
    from .runner import main as run
    return run(args)
