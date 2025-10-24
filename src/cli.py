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
from src.prompts import generate_prompt_template, save_transformed_guideline, transform_guideline_csv


console = Console()


app = typer.Typer(
    name="car-contamination-classifier",
    help="Car contamination classification using Vision Language Models",
    add_completion=False,
)


@app.command("infer")
def single_inference(
    image: Annotated[Path, typer.Argument(help="Path to image file")],
    prompt: Annotated[Path, typer.Argument(help="Path to prompt file")],
    api_url: Annotated[
        str,
        typer.Option(help="VLM API endpoint URL"),
    ] = "http://vllm.mlops.socarcorp.co.kr/v1/chat/completions",
    model: Annotated[str, typer.Option(help="Model name")] = "qwen3-vl-8b-instruct",
    max_tokens: Annotated[int, typer.Option(help="Maximum tokens to generate")] = 1000,
    temperature: Annotated[float, typer.Option(help="Sampling temperature")] = 0.0,
    output: Annotated[Path | None, typer.Option(help="Output file path (optional)")] = None,
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

    # Initialize client
    typer.echo(f"API URL: {api_url}")
    typer.echo(f"Model: {model}")
    typer.echo(f"Image: {image}")
    typer.echo()

    client = VLMClient(api_url=api_url, model=model)

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
    output: Annotated[Path | None, typer.Argument(help="Output CSV file path")] = None,
    api_url: Annotated[
        str,
        typer.Option(help="VLM API endpoint URL"),
    ] = "http://vllm.mlops.socarcorp.co.kr/v1/chat/completions",
    model: Annotated[str, typer.Option(help="Model name")] = "qwen3-vl-8b-instruct",
    max_tokens: Annotated[int, typer.Option(help="Maximum tokens to generate")] = 1000,
    temperature: Annotated[float, typer.Option(help="Sampling temperature")] = 0.0,
    limit: Annotated[int | None, typer.Option(help="Maximum number of images to process (default: all)")] = None,
    max_workers: Annotated[int, typer.Option(help="Number of parallel workers (default: 16)")] = 16,
    enable_langfuse: Annotated[
        bool, typer.Option("--enable-langfuse/--no-langfuse", help="Enable Langfuse monitoring")
    ] = True,
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

    if output is None:
        output_str = Prompt.ask("[cyan]💾 Output CSV file path[/cyan]")
        output = Path(output_str)

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
    config_table.add_row("📊 Langfuse", "[green]Enabled[/green]" if enable_langfuse else "[red]Disabled[/red]")
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
            enable_langfuse=enable_langfuse,
        )

        # Display summary in a table
        console.print()
        summary_table = Table(title="📊 Summary", show_header=False, box=None)
        summary_table.add_column("Key", style="cyan", width=25)
        summary_table.add_column("Value", style="white")

        summary_table.add_row("🖼️  Total images", str(summary["total"]))
        summary_table.add_row("✅ Successful", f"[green]{summary['successful']}[/green]")
        summary_table.add_row("❌ Failed", f"[red]{summary['failed']}[/red]")
        summary_table.add_row("⏱️  Total time", f"[bold yellow]{summary['total_time']:.2f}s[/bold yellow]")
        summary_table.add_row("📊 Time per image (avg)", f"[bold cyan]{summary['avg_time_per_image']:.2f}s[/bold cyan]")
        summary_table.add_row("⚡ API latency (avg)", f"{summary['avg_latency']:.3f}s")
        summary_table.add_row("💾 Results saved to", str(summary["output_path"]))

        console.print(summary_table)
        console.print()

        # Calculate and display speedup information
        speedup = summary["avg_latency"] / summary["avg_time_per_image"] if summary["avg_time_per_image"] > 0 else 1
        speedup_info = f"🚀 Parallel speedup: [bold green]{speedup:.1f}x faster[/bold green] than sequential processing"
        console.print(speedup_info)
        console.print()

        console.print(Panel.fit("✓ Batch inference completed successfully!", style="bold green"))

    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1) from e


@app.command("dashboard")
def launch_dashboard(
    port: Annotated[int, typer.Option(help="Port to run the dashboard on")] = 8501,
):
    """Launch the Streamlit dashboard for visualizing inference results."""
    import subprocess
    import sys

    console.print(Panel.fit("📊 Launching Streamlit Dashboard", style="bold magenta"))
    console.print()
    console.print(f"[cyan]🌐 Dashboard will be available at:[/cyan] [green]http://localhost:{port}[/green]")
    console.print("[cyan]📂 Default CSV path:[/cyan] results/inference_results.csv")
    console.print("[cyan]📁 Default images path:[/cyan] images/sample_images/images")
    console.print()
    console.print("[yellow]Press Ctrl+C to stop the server[/yellow]")
    console.print()

    try:
        # Run streamlit
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "src/dashboard/app.py", "--server.port", str(port)],
            check=True,
        )
    except KeyboardInterrupt:
        console.print()
        console.print(Panel.fit("✓ Dashboard stopped", style="bold green"))
    except subprocess.CalledProcessError as e:
        console.print(f"[red]❌ Error running dashboard: {e}[/red]")
        raise typer.Exit(1) from e


@app.command("generate-prompt")
def generate_prompt(
    guideline: Annotated[Path, typer.Argument(help="Path to guideline CSV file")],
    output: Annotated[Path | None, typer.Argument(help="Output prompt file path (optional)")] = None,
    save_transformed: Annotated[
        bool, typer.Option("--save-transformed", help="Save transformed guideline CSV")
    ] = False,
    template_version: Annotated[int, typer.Option("--template-version", help="Template version to use")] = 1,
):
    """Generate prompt template from guideline CSV."""
    typer.echo("=" * 60)
    typer.echo("Prompt Generation")
    typer.echo("=" * 60)

    # Validate guideline path
    if not guideline.exists():
        typer.echo(f"Error: Guideline file not found: {guideline}", err=True)
        raise typer.Exit(1)

    try:
        # Step 1: Transform guideline
        typer.echo("\nStep 1: Transforming guideline CSV")
        typer.echo("-" * 60)
        transformed_rows = transform_guideline_csv(guideline)
        typer.echo(f"Transformed {len(transformed_rows)} guideline entries")

        # Save transformed guideline if requested
        if save_transformed:
            transformed_output = guideline.parent / f"{guideline.stem}_transformed.csv"
            save_transformed_guideline(transformed_rows, transformed_output)
            typer.echo(f"Transformed guideline saved to: {transformed_output}")

        # Step 2: Generate prompt
        typer.echo("\nStep 2: Generating prompt template")
        typer.echo("-" * 60)
        typer.echo(f"Using template version: {template_version}")
        prompt = generate_prompt_template(transformed_rows, template_version=template_version)
        typer.echo("Prompt generated successfully")

        # Determine output path
        if output is None:
            # Auto-generate version number
            prompts_dir = guideline.parent.parent / "prompts"
            prompts_dir.mkdir(exist_ok=True)
            existing_versions = list(prompts_dir.glob("car_contamination_classification_prompt_v*.txt"))
            if existing_versions:
                # Extract version numbers
                versions = []
                for p in existing_versions:
                    try:
                        version = int(p.stem.split("_v")[1])
                        versions.append(version)
                    except (IndexError, ValueError):
                        continue
                next_version = max(versions) + 1 if versions else 1
            else:
                next_version = 1
            output = prompts_dir / f"car_contamination_classification_prompt_v{next_version}.txt"

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

        if save_transformed:
            typer.echo("\nGenerated files:")
            typer.echo(f"  - Transformed guideline: {transformed_output}")
            typer.echo(f"  - Prompt template: {output}")
        else:
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
