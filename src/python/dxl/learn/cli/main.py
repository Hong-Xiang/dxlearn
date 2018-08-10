import click


class CLI(click.MultiCommand):
    commands = {'zoo': None}

    def __init__(self):
        super().__init__(name='dxlearn',
                         help='Machine learning CLI.')

    def list_commands(self, ctx):
        return sorted(self.commands.keys())

    def get_command(self, ctx, name):
        from dxl.learn.zoo.cli import zoo
        if name in self.commands:
            if self.commands[name] is None:
                mapping = {'zoo': zoo}
                self.commands[name] = mapping.get(name)
        return self.commands.get(name)


dxlearn = CLI()
