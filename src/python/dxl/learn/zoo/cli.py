import click


class CLI(click.MultiCommand):
    commands = {'incident': None}

    def __init__(self):
        super().__init__(name='zoo', help='ML zoo CLI.')

    def list_commands(self, ctx):
        return sorted(self.commands.keys())

    def get_command(self, ctx, name):
        from dxl.learn.zoo.incident.cli import incident
        if name in self.commands:
            if self.commands[name] is None:
                mapping = {'incident': incident}
                self.commands[name] = mapping.get(name)
        return self.commands.get(name)


zoo = CLI()

if __name__ == "__main__":
    zoo()