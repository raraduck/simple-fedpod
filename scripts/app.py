import logging

log = logging.getLogger(__name__)


class App:
    def __init__(self, args):
        self.args = args

    def run(self):
        pass


def main():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    App(args).run()


if __name__ == '__main__':
    main()
