"""
`pa admin` subcommand — admin-only operations.

Only you (the admin) use these. Regular users never see this.

Commands:
  pa admin init                          Create the registry (first time)
  pa admin set-gpu http://<IP>:9000/v1   Set the current GPU server URL
  pa admin deactivate                    Mark GPU as offline
  pa admin status                        Check registry state
"""

import sys
import click
from pathlib import Path
from rich.console import Console

console = Console()


@click.group("admin")
def admin():
    """Admin commands — manage the GPU registry.

    \b
    COMMANDS:
      pa admin init                       Create the registry (first time only)
      pa admin set-gpu http://<IP>:9000/v1  Update GPU server URL
      pa admin deactivate                 Mark GPU as offline
      pa admin status                     Check registry state
    """
    pass


@admin.command("init")
def admin_init():
    """Create the registry Gist (run this once, ever)."""
    from cli.banner import print_mini_banner
    print_mini_banner(console)

    from cli.config_manager import load_config, set_value
    from cli.registry import update_registry, _get_gist_id

    if _get_gist_id():
        console.print("[yellow]Registry already exists.[/yellow]")
        console.print("[dim]Run 'pa admin set-gpu <URL>' to update the server URL.[/dim]")
        return

    cfg = load_config()
    token = cfg.get("github_token", "")
    if not token:
        console.print("[red]Error:[/red] No GitHub token. Run: pa config set --token ghp_...")
        sys.exit(1)

    console.print("[bold]Creating registry...[/bold]")
    result = update_registry(
        server_url="",
        active=False,
        message="GPU not yet configured. Admin needs to run: pa admin set-gpu <URL>",
        github_token=token,
    )

    if result["ok"]:
        gist_id = result["gist_id"]
        set_value("registry_gist_id", gist_id)

        # Auto-patch the hardcoded gist ID in registry.py
        _patch_registry_source(gist_id)

        console.print(f"\n[green]✓ Registry created![/green]")
        console.print(f"  Gist ID: [cyan]{gist_id}[/cyan]")
        console.print(f"\n[bold]What happened:[/bold]")
        console.print(f"  • Created a private GitHub Gist as the server registry")
        console.print(f"  • Auto-updated [cyan]cli/registry.py[/cyan] with the Gist ID")
        console.print(f"\n[bold]Next steps:[/bold]")
        console.print(f"  1. Commit the change: [cyan]git add cli/registry.py && git commit -m 'Add registry'[/cyan]")
        console.print(f"  2. When you activate a GPU: [cyan]pa admin set-gpu http://<GPU_IP>:9000/v1[/cyan]")
        console.print(f"\n[dim]Users only need: pa config set --key <KEY> --token <TOKEN>[/dim]")
    else:
        console.print(f"[red]Error:[/red] {result['error']}")
        sys.exit(1)


def _patch_registry_source(gist_id: str):
    """Write the gist ID into cli/registry.py so it's baked into the package."""
    import re
    registry_path = Path(__file__).parent.parent / "registry.py"

    if not registry_path.exists():
        return

    content = registry_path.read_text()
    # Replace the HARDCODED_GIST_ID line
    new_content = re.sub(
        r'HARDCODED_GIST_ID = ".*?"',
        f'HARDCODED_GIST_ID = "{gist_id}"',
        content,
    )
    registry_path.write_text(new_content)


@admin.command("set-gpu")
@click.argument("server_url")
@click.option("--message", "-m", type=str, default="",
              help="Optional message shown to users (e.g. 'Available until 6pm').")
def admin_set_gpu(server_url, message):
    """Activate a GPU and set the server URL.

    \b
    EXAMPLE:
      pa admin set-gpu http://134.199.194.54:9000/v1
      pa admin set-gpu http://134.199.194.54:9000/v1 -m "Available until 6pm"
    """
    from cli.banner import print_mini_banner
    print_mini_banner(console)

    from cli.registry import update_registry

    console.print(f"[bold]Updating registry...[/bold]")
    console.print(f"  Server: {server_url}")

    result = update_registry(server_url=server_url, active=True, message=message)

    if result["ok"]:
        console.print(f"\n[green]✓ GPU activated![/green]")
        console.print(f"  All users will now connect to: [cyan]{server_url}[/cyan]")
        if message:
            console.print(f"  Message: {message}")
    else:
        console.print(f"\n[red]Error:[/red] {result['error']}")
        sys.exit(1)


@admin.command("deactivate")
@click.option("--message", "-m", type=str, default="GPU not active. Try again later.",
              help="Message shown to users.")
def admin_deactivate(message):
    """Mark the GPU as offline. Users will see a 'try later' message."""
    from cli.banner import print_mini_banner
    print_mini_banner(console)

    from cli.registry import update_registry

    result = update_registry(server_url="", active=False, message=message)

    if result["ok"]:
        console.print(f"[green]✓ GPU deactivated.[/green]")
        console.print(f"  Users will see: [dim]{message}[/dim]")
    else:
        console.print(f"[red]Error:[/red] {result['error']}")
        sys.exit(1)


@admin.command("status")
def admin_status():
    """Check the current registry state."""
    from cli.banner import print_mini_banner
    print_mini_banner(console)

    from cli.registry import fetch_server_url, _get_gist_id

    gist_id = _get_gist_id()
    if not gist_id:
        console.print("[yellow]No registry configured.[/yellow]")
        console.print("[dim]Run 'pa admin init' first.[/dim]")
        return

    console.print(f"[bold]Registry:[/bold] Gist {gist_id[:12]}...")

    info = fetch_server_url()

    if info["active"]:
        console.print(f"  [green]✓ GPU ACTIVE[/green]")
        console.print(f"  Server: [cyan]{info['server_url']}[/cyan]")
    else:
        console.print(f"  [red]✗ GPU OFFLINE[/red]")

    if info["message"]:
        console.print(f"  Message: {info['message']}")

    if info["error"]:
        console.print(f"  [red]Error: {info['error']}[/red]")
