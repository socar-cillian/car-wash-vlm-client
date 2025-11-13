"""Integrated CLI for car contamination classification."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from src.api import VLMClient
from src.inference import run_batch_inference
from src.prompts import generate_prompt_template, parse_guideline_v2


console = Console()


app = typer.Typer(
    name="car-contamination-classifier",
    help="Car contamination classification using Vision Language Models",
    add_completion=False,
)


def _get_api_url(internal: bool, model: str) -> str:
    """
    Get the appropriate API URL based on whether running internally or externally.

    Args:
        internal: Whether running inside Kubernetes cluster
        model: Model name to determine the service

    Returns:
        Full API URL for the VLM service
    """
    if internal:
        # Internal Kubernetes service URL format: <service_name>.<namespace>.svc.cluster.local
        # Default to qwen3-vl-8b for now
        service_name = "vllm-qwen3-vl-8b-engine-service"
        namespace = "vllm"
        return f"http://{service_name}.{namespace}.svc.cluster.local:8000/v1/chat/completions"
    else:
        # External URL
        return "https://vllm.mlops.socarcorp.co.kr/v1/chat/completions"


@app.command("infer")
def single_inference(
    image: Annotated[Path, typer.Argument(help="Path to image file")],
    prompt: Annotated[Path, typer.Argument(help="Path to prompt file")],
    api_url: Annotated[
        str | None,
        typer.Option(help="VLM API endpoint URL (overrides --internal)"),
    ] = None,
    model: Annotated[str, typer.Option(help="Model name")] = "qwen3-vl-8b-instruct",
    max_tokens: Annotated[int, typer.Option(help="Maximum tokens to generate")] = 1000,
    temperature: Annotated[float, typer.Option(help="Sampling temperature")] = 0.0,
    output: Annotated[Path | None, typer.Option(help="Output file path (optional)")] = None,
    internal: Annotated[bool, typer.Option("--internal", help="Use internal Kubernetes service URL")] = False,
):
    """Run inference on a single image."""
    typer.echo("=" * 60)
    typer.echo("Single Image Inference")
    typer.echo("=" * 60)

    # Validate paths
    if not image.exists():
        typer.echo(f"Error: Image file not found: {image}", err=True)
        raise typer.Exit(1)

    if not prompt.exists():
        typer.echo(f"Error: Prompt file not found: {prompt}", err=True)
        raise typer.Exit(1)

    # Load prompt
    with open(prompt, encoding="utf-8") as f:
        prompt_text = f.read()

    # Determine API URL
    if api_url is None:
        api_url = _get_api_url(internal, model)

    # Initialize client
    typer.echo(f"API URL: {api_url}")
    typer.echo(f"Model: {model}")
    typer.echo(f"Image: {image}")
    typer.echo()

    client = VLMClient(api_url=api_url, model=model)

    # Check server health
    typer.echo("Checking server health...")
    health_result = client.check_health(timeout=10)

    if not health_result["healthy"]:
        typer.echo(f"✗ Server health check failed: {health_result['error']}", err=True)
        raise typer.Exit(1)

    typer.echo(f"✓ Server is healthy (response time: {health_result['response_time']:.2f}s)")
    if health_result.get("endpoint"):
        typer.echo(f"  Health endpoint: {health_result['endpoint']}")
    typer.echo()

    # Run inference
    typer.echo("Running inference...")
    try:
        response = client.query(
            image_input=str(image),
            prompt_input=prompt_text,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Extract result
        if "choices" in response and len(response["choices"]) > 0:
            result = response["choices"][0]["message"]["content"]

            # Save or print result
            if output:
                output.parent.mkdir(parents=True, exist_ok=True)
                with open(output, "w", encoding="utf-8") as f:
                    f.write(result)
                typer.echo(f"\n✓ Result saved to: {output}")
            else:
                typer.echo("\nResult:")
                typer.echo("-" * 60)
                typer.echo(result)

            typer.echo("\n" + "=" * 60)
            typer.echo("✓ Inference completed successfully!")
            typer.echo("=" * 60)
        else:
            typer.echo("Error: No response from model", err=True)
            raise typer.Exit(1)

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from e


@app.command("batch-infer")
def batch_inference(
    input_csv: Annotated[Path | None, typer.Argument(help="Input CSV file with file_name and GT columns")] = None,
    images_dir: Annotated[Path | None, typer.Argument(help="Directory containing images")] = None,
    prompt: Annotated[Path | None, typer.Argument(help="Path to prompt file")] = None,
    api_url: Annotated[
        str | None,
        typer.Option(help="VLM API endpoint URL (overrides --internal)"),
    ] = None,
    model: Annotated[str, typer.Option(help="Model name")] = "qwen3-vl-8b-instruct",
    max_tokens: Annotated[int, typer.Option(help="Maximum tokens to generate")] = 1000,
    temperature: Annotated[float, typer.Option(help="Sampling temperature")] = 0.0,
    limit: Annotated[int | None, typer.Option(help="Maximum number of images to process (default: all)")] = None,
    max_workers: Annotated[int, typer.Option(help="Number of parallel workers (default: 4)")] = 4,
    internal: Annotated[bool, typer.Option("--internal", help="Use internal Kubernetes service URL")] = False,
):
    """Run batch inference on multiple images specified in CSV file."""
    console.print(Panel.fit("🚗 Batch Inference", style="bold magenta"))

    # Interactive input if arguments not provided
    if input_csv is None:
        console.print()
        input_csv_str = Prompt.ask("[cyan]📄 Input CSV file path[/cyan]")
        input_csv = Path(input_csv_str)

    if images_dir is None:
        images_dir_str = Prompt.ask("[cyan]📁 Images directory path[/cyan]")
        images_dir = Path(images_dir_str)

    if prompt is None:
        prompt_str = Prompt.ask("[cyan]📝 Prompt file path[/cyan]")
        prompt = Path(prompt_str)

    # Ask for limit if not provided via CLI and in interactive mode
    if limit is None:
        limit_str = Prompt.ask("[cyan]🔢 Maximum number of images to process (press Enter for all)[/cyan]", default="")
        if limit_str and limit_str.strip():
            try:
                limit = int(limit_str)
            except ValueError:
                console.print("[yellow]⚠️  Invalid number, processing all images[/yellow]")
                limit = None

    # Ask for internal mode if api_url is not provided
    if api_url is None and not internal:
        internal_str = Prompt.ask("[cyan]🔧 Running inside Kubernetes cluster? (y/N)[/cyan]", default="N")
        internal = internal_str.strip().lower() in ["y", "yes"]

    console.print()

    # Validate paths
    if not input_csv.exists():
        console.print(f"[red]❌ Error: Input CSV file not found: {input_csv}[/red]")
        raise typer.Exit(1)

    if not images_dir.exists():
        console.print(f"[red]❌ Error: Images directory not found: {images_dir}[/red]")
        raise typer.Exit(1)

    if not prompt.exists():
        console.print(f"[red]❌ Error: Prompt file not found: {prompt}[/red]")
        raise typer.Exit(1)

    # Auto-generate output path: results/{prompt_name}_result.csv
    prompt_name = prompt.stem  # e.g., "promptv4" from "promptv4.txt"
    output = Path("results") / f"{prompt_name}_result.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    # Determine API URL
    if api_url is None:
        api_url = _get_api_url(internal, model)

    # Display configuration in a table
    config_table = Table(title="⚙️  Configuration", show_header=False, box=None)
    config_table.add_column("Key", style="cyan", width=20)
    config_table.add_column("Value", style="white")

    config_table.add_row("📄 Input CSV", str(input_csv))
    config_table.add_row("📁 Images directory", str(images_dir))
    config_table.add_row("📝 Prompt file", str(prompt))
    config_table.add_row("💾 Output CSV", str(output))
    config_table.add_row("🌐 API URL", api_url)
    config_table.add_row("🤖 Model", model)
    config_table.add_row("⚡ Workers", str(max_workers))
    if limit:
        config_table.add_row("🔢 Limit", str(limit))

    console.print(config_table)
    console.print()

    try:
        summary = run_batch_inference(
            input_csv=input_csv,
            images_dir=images_dir,
            prompt_path=prompt,
            output_csv=output,
            api_url=api_url,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            limit=limit,
            max_workers=max_workers,
        )

        # Display summary in a table
        console.print()
        summary_table = Table(title="📊 Summary", show_header=False, box=None)
        summary_table.add_column("Key", style="cyan", width=25)
        summary_table.add_column("Value", style="white")

        # Calculate throughput
        throughput = summary["total"] / summary["total_time"] if summary["total_time"] > 0 else 0
        speedup = summary["avg_latency"] / summary["avg_time_per_image"] if summary["avg_time_per_image"] > 0 else 1

        summary_table.add_row("🖼️  Total images", str(summary["total"]))
        summary_table.add_row("✅ Successful", f"[green]{summary['successful']}[/green]")
        summary_table.add_row("❌ Failed", f"[red]{summary['failed']}[/red]")
        summary_table.add_row("⏱️  Total time", f"[bold yellow]{summary['total_time']:.2f}s[/bold yellow]")
        summary_table.add_row("🚀 Throughput", f"[bold magenta]{throughput:.2f} images/sec[/bold magenta]")
        summary_table.add_row("📊 Time per image (avg)", f"[bold cyan]{summary['avg_time_per_image']:.2f}s[/bold cyan]")
        summary_table.add_row("⚡ API latency (avg)", f"{summary['avg_latency']:.3f}s")
        summary_table.add_row("⚙️  Parallel speedup", f"[bold green]{speedup:.1f}x[/bold green]")
        summary_table.add_row("💾 Results saved to", str(summary["output_path"]))

        console.print(summary_table)
        console.print()

        # Show appropriate message based on results
        if summary["successful"] == 0 and summary["failed"] > 0:
            console.print(Panel.fit("❌ All inference requests failed!", style="bold red"))
            console.print()

            # Analyze errors to provide helpful diagnostics
            import pandas as pd

            results_df = pd.read_csv(summary["output_path"])
            error_samples = results_df[~results_df["success"]]["error"].head(3).tolist()

            # Check for common error patterns
            has_403_error = any("403" in str(err) for err in error_samples if err)
            has_500_error = any("500" in str(err) for err in error_samples if err)

            if has_403_error:
                console.print("[red]❌ Server cannot access image URLs (403 Forbidden)[/red]")
                console.print()
                console.print("[yellow]Possible solutions:[/yellow]")
                console.print("  • Check S3 bucket permissions - server may need access")
                console.print("  • Make S3 bucket public or add server IP to allowlist")
                console.print("  • Use signed URLs if images are in private bucket")
                console.print("  • Download images locally and use local paths instead")
            elif has_500_error:
                console.print("[red]Server returned 500 errors. Common issues:[/red]")
                console.print("  • Server out of memory or overloaded")
                console.print("  • Invalid image formats")
                console.print("  • Check server logs for details")
            else:
                console.print("[red]All images failed to process. Common issues:[/red]")
                console.print("  • Invalid image URLs or formats")
                console.print("  • Network connectivity issues")
                console.print("  • Server configuration problems")

            console.print()
            console.print("[cyan]Sample errors:[/cyan]")
            for i, err in enumerate(error_samples[:2], 1):
                if err and len(str(err)) > 100:
                    console.print(f"  {i}. {str(err)[:100]}...")
                elif err:
                    console.print(f"  {i}. {err}")
            console.print()
            console.print(f"[yellow]Check the error column in {summary['output_path']} for full details[/yellow]")
            raise typer.Exit(1)
        elif summary["failed"] > 0:
            success_rate = (summary["successful"] / summary["total"]) * 100
            msg = (
                f"⚠️  Batch inference completed with {summary['failed']} failures " f"({success_rate:.1f}% success rate)"
            )
            console.print(Panel.fit(msg, style="bold yellow"))
        else:
            console.print(Panel.fit("✓ Batch inference completed successfully!", style="bold green"))

    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1) from e


@app.command("generate-prompt")
def generate_prompt(
    guideline: Annotated[Path, typer.Argument(help="Path to guideline v2 CSV file")],
    output: Annotated[Path | None, typer.Argument(help="Output prompt file path (optional)")] = None,
    template_version: Annotated[int, typer.Option("--template-version", help="Template version to use")] = 1,
):
    """Generate prompt template from guideline v2 CSV."""
    typer.echo("=" * 60)
    typer.echo("Prompt Generation")
    typer.echo("=" * 60)

    # Validate guideline path
    if not guideline.exists():
        typer.echo(f"Error: Guideline file not found: {guideline}", err=True)
        raise typer.Exit(1)

    try:
        # Step 1: Parse guideline v2
        typer.echo("\nStep 1: Parsing guideline v2 CSV")
        typer.echo("-" * 60)
        parsed_rows = parse_guideline_v2(guideline)
        typer.echo(f"Parsed {len(parsed_rows)} guideline entries")

        # Step 2: Generate prompt
        typer.echo("\nStep 2: Generating prompt template")
        typer.echo("-" * 60)
        typer.echo(f"Using template version: {template_version}")
        prompt = generate_prompt_template(parsed_rows, template_version=template_version)
        typer.echo("Prompt generated successfully")

        # Determine output path based on guideline version
        if output is None:
            # Extract version from guideline filename (e.g., guideline_v2.csv -> v2)
            guideline_stem = guideline.stem  # e.g., "guideline_v2"
            version_str = guideline_stem.split("_v")[1] if "_v" in guideline_stem else "1"

            prompts_dir = guideline.parent.parent / "prompts"
            prompts_dir.mkdir(exist_ok=True)
            output = prompts_dir / f"prompt_v{version_str}.txt"

        # Step 3: Save prompt
        typer.echo("\nStep 3: Saving prompt")
        typer.echo("-" * 60)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            f.write(prompt)
        typer.echo(f"Prompt saved to: {output}")

        typer.echo("\n" + "=" * 60)
        typer.echo("✓ Prompt generation completed successfully!")
        typer.echo("=" * 60)
        typer.echo(f"\nGenerated file: {output}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        import traceback

        traceback.print_exc()
        raise typer.Exit(1) from e


def main():
    """Entry point for CLI."""
    import sys

    # Show help if no arguments provided
    if len(sys.argv) == 1:
        sys.argv.append("--help")

    app()


if __name__ == "__main__":
    main()
