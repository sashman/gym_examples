"""Entry point for gym_examples."""

from gym_examples.cli import main  # pragma: no cover
import typer

app = typer.Typer(pretty_exceptions_enable=False)

# Register the main command
app.command()(main)

if __name__ == "__main__":  # pragma: no cover
    app(prog_name="gym-examples")