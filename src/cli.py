"""Integrated CLI for car contamination classification."""

import sys
import time
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from src.api import VLMClient
from src.inference import run_batch_inference, run_gemini_batch_inference
from src.prompts import generate_prompt, parse_guideline


# Remove default logger to avoid duplicate console output
logger.remove()


console = Console()

# Default base path for Kubernetes environment
DEFAULT_BASE_PATH = Path("/home/jovyan")


def _resolve_path(path_str: str) -> Path:
    """
    Resolve path with fallback to DEFAULT_BASE_PATH.

    If the path doesn't exist, try prepending DEFAULT_BASE_PATH.

    Args:
        path_str: Path string from user input

    Returns:
        Resolved Path object
    """
    path = Path(path_str)
    if path.exists():
        return path

    # Try with default base path
    alt_path = DEFAULT_BASE_PATH / path_str.lstrip("/")
    if alt_path.exists():
        return alt_path

    # Return original path (will fail validation later with proper error)
    return path


app = typer.Typer(
    name="car-contamination-classifier",
    help="Car contamination classification using Vision Language Models",
    add_completion=False,
)


def _get_api_url(internal: bool, model: str, namespace: str = "vllm-test") -> str:
    """
    Get the appropriate API URL based on whether running internally or externally.

    Args:
        internal: Whether running inside Kubernetes cluster
        model: Model name to determine the service
        namespace: Kubernetes namespace (vllm or vllm-test)

    Returns:
        Full API URL for the VLM service
    """
    # Service name pattern: <release>-<model>-engine-service
    # Release name matches namespace (vllm or vllm-test)
    # Model name: qwen3-vl-8b
    service_name = f"{namespace}-qwen3-vl-8b-engine-service"

    if internal:
        # Internal K8s URL
        return f"http://{service_name}.{namespace}.svc.cluster.local:8000/v1/chat/completions"
    else:
        # External URL - model name is used as subdomain (from ingress-engine.yaml)
        return "https://qwen3-vl-8b.mlops.socarcorp.co.kr/v1/chat/completions"


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
) -> None:
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
        api_url = _get_api_url(internal, model, "vllm-test")

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
    max_tokens: Annotated[
        int | None, typer.Option(help="Maximum tokens to generate (default: server's max_model_len)")
    ] = None,
    temperature: Annotated[float, typer.Option(help="Sampling temperature")] = 0.0,
    limit: Annotated[int | None, typer.Option(help="Maximum number of images to process (default: all)")] = None,
    max_workers: Annotated[
        int | None, typer.Option(help="Number of parallel workers (default: auto based on server replicas)")
    ] = None,
    workers_per_replica: Annotated[int, typer.Option(help="Workers per GPU replica for auto-scaling (default: 4)")] = 4,
    internal: Annotated[bool, typer.Option("--internal", help="Use internal Kubernetes service URL")] = False,
) -> None:
    """Run batch inference on multiple images specified in CSV file."""
    console.print(Panel.fit("🚗 Batch Inference", style="bold magenta"))

    # Interactive input if arguments not provided
    if input_csv is None:
        console.print()
        input_csv_str = Prompt.ask("[cyan]📄 Input CSV file path[/cyan]")
        input_csv = _resolve_path(input_csv_str)

    if images_dir is None:
        images_dir_str = Prompt.ask("[cyan]📁 Images directory path[/cyan]")
        images_dir = _resolve_path(images_dir_str)

    if prompt is None:
        prompt_str = Prompt.ask("[cyan]📝 Prompt file path[/cyan]")
        prompt = _resolve_path(prompt_str)

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
    namespace = "vllm-test"  # default
    if api_url is None and not internal:
        internal_str = Prompt.ask("[cyan]🔧 Running inside Kubernetes cluster? (y/N)[/cyan]", default="N")
        internal = internal_str.strip().lower() in ["y", "yes"]

    # Ask for namespace if api_url is not provided
    if api_url is None:
        namespace = Prompt.ask(
            "[cyan]🏷️  Select namespace (vllm / vllm-test)[/cyan]", choices=["vllm", "vllm-test"], default="vllm-test"
        )

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

    # Auto-generate output path: results/{dataset_name}_{prompt_name}_result.csv
    prompt_name = prompt.stem  # e.g., "prompt_v4" from "prompt_v4.txt"
    # Extract dataset name from images_dir (parent folder of 'images' or the folder itself)
    dataset_name = images_dir.parent.name if images_dir.name == "images" else images_dir.name
    output = Path("results") / f"{dataset_name}_{prompt_name}_result.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    # Setup logging - save to same location as output CSV with .log extension
    log_path = output.with_suffix(".log")
    logger.add(
        log_path,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="INFO",
        rotation=None,  # No rotation for single run
        mode="w",  # Overwrite existing log
    )
    logger.info(f"Logging to {log_path}")

    # Determine API URL
    if api_url is None:
        api_url = _get_api_url(internal, model, namespace)

    # Check server health with cold start support (KEDA may have scaled to 0)
    console.print(f"[cyan]🌐 API URL: {api_url}[/cyan]")
    console.print("Checking server health...")
    temp_client = VLMClient(api_url=api_url, model=model)
    health_result = temp_client.check_health(timeout=10)

    if not health_result["healthy"]:
        # Server might be scaled to 0, wait for cold start
        console.print("[yellow]⏳ Server not ready (may be scaled to zero). Waiting for cold start...[/yellow]")
        console.print("[yellow]   This can take 5-10 minutes for GPU pod startup + model loading[/yellow]")
        logger.info("Server not ready, waiting for cold start...")

        # Wait up to 15 minutes for cold start (GPU startup + model download + loading)
        max_wait = 900  # 15 minutes
        wait_interval = 30  # Check every 30 seconds
        elapsed = 0

        with console.status("[yellow]Waiting for server...[/yellow]") as status:
            while elapsed < max_wait:
                time.sleep(wait_interval)
                elapsed += wait_interval
                health_result = temp_client.check_health(timeout=30)

                if health_result["healthy"]:
                    console.print(f"\n[green]✓ Server is now ready after {elapsed}s![/green]")
                    logger.info(f"Server ready after {elapsed}s cold start")
                    break

                status.update(f"[yellow]Waiting for server... ({elapsed}s / {max_wait}s)[/yellow]")
                logger.info(f"Still waiting for server... {elapsed}s elapsed")
            else:
                console.print(f"\n[red]❌ Server did not become ready after {max_wait}s[/red]")
                console.print("[red]   Check if GPU nodes are available and KEDA is working[/red]")
                raise typer.Exit(1)
    else:
        console.print(f"[green]✓ Server is healthy (response time: {health_result['response_time']:.2f}s)[/green]")
        if health_result.get("endpoint"):
            console.print(f"[dim]  Health endpoint: {health_result['endpoint']}[/dim]")

    console.print()

    # Get max_model_len from server for display
    model_info = temp_client.get_model_info()
    server_max_model_len = model_info.get("max_model_len") if model_info else None

    # Set workers based on pod count
    # Currently using 1 pod with 4 workers per pod
    if max_workers is None:
        max_workers = 4  # Fixed: 1 pod × 4 workers per pod
        console.print(f"[cyan]ℹ️  Using {max_workers} workers (1 pod × 4 workers)[/cyan]")
        console.print()
        logger.info(f"Using {max_workers} workers")

    # If max_tokens not specified, use a reasonable default (not max_model_len!)
    # max_model_len is total context (input + output), so we can't use it all for output
    if max_tokens is None:
        max_tokens = 1000  # reasonable default for output tokens
        console.print(f"[cyan]ℹ️  Using default max_tokens: {max_tokens}[/cyan]")
        console.print()

    # Log configuration
    logger.info("=" * 60)
    logger.info("BATCH INFERENCE STARTED")
    logger.info("=" * 60)
    logger.info(f"Input CSV: {input_csv}")
    logger.info(f"Images directory: {images_dir}")
    logger.info(f"Prompt file: {prompt}")
    logger.info(f"Output CSV: {output}")
    logger.info(f"API URL: {api_url}")
    logger.info(f"Model: {model}")
    if server_max_model_len:
        logger.info(f"Max Model Len: {server_max_model_len}")
    logger.info(f"Max Tokens: {max_tokens}")
    logger.info(f"Workers: {max_workers}")
    if limit:
        logger.info(f"Limit: {limit}")
    logger.info("-" * 60)

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
    if server_max_model_len:
        config_table.add_row("📏 Max Model Len", f"{server_max_model_len} (input + output)")
    config_table.add_row("🔢 Max Tokens", f"{max_tokens} (output)")
    config_table.add_row("⚡ Workers", str(max_workers))
    if limit:
        config_table.add_row("📊 Limit", str(limit))

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

        # Log summary
        logger.info("=" * 60)
        logger.info("BATCH INFERENCE COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Total images: {summary['total']}")
        logger.info(f"Successful: {summary['successful']}")
        logger.info(f"Failed: {summary['failed']}")
        logger.info(f"Total time: {summary['total_time']:.2f}s")
        logger.info(f"Throughput: {throughput:.2f} images/sec")
        logger.info(f"Time per image (avg): {summary['avg_time_per_image']:.2f}s")
        logger.info(f"API latency (avg): {summary['avg_latency']:.3f}s")
        logger.info(f"Parallel speedup: {speedup:.1f}x")
        logger.info(f"Results saved to: {summary['output_path']}")
        logger.info(f"Log saved to: {log_path}")
        logger.info("=" * 60)

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
            msg = f"⚠️  Batch inference completed with {summary['failed']} failures ({success_rate:.1f}% success rate)"
            console.print(Panel.fit(msg, style="bold yellow"))
        else:
            console.print(Panel.fit("✓ Batch inference completed successfully!", style="bold green"))

    except Exception as e:
        logger.error(f"Batch inference failed with error: {e}")
        console.print(f"[red]❌ Error: {e}[/red]")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1) from e


@app.command("gemini-batch")
def gemini_batch_inference(
    input_csv: Annotated[
        Path | None, typer.Argument(help="Input CSV file with file_name column (optional for GCS)")
    ] = None,
    images_dir: Annotated[str | None, typer.Argument(help="Images path (local dir or GCS: gs://bucket/path)")] = None,
    prompt: Annotated[Path | None, typer.Argument(help="Path to prompt file")] = None,
    model: Annotated[str | None, typer.Option(help="Model name (default: from .env MODEL_NAME)")] = None,
    max_tokens: Annotated[int, typer.Option(help="Maximum tokens to generate")] = 1000,
    temperature: Annotated[float, typer.Option(help="Sampling temperature")] = 0.0,
    limit: Annotated[int | None, typer.Option(help="Maximum number of images to process (default: all)")] = None,
    poll_interval: Annotated[int, typer.Option(help="Seconds between status checks")] = 30,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip cost confirmation prompt")] = False,
    gcs_bucket: Annotated[
        str | None,
        typer.Option("--gcs-bucket", help="GCS bucket name for Vertex AI batch jobs"),
    ] = None,
) -> None:
    """Run batch inference using Gemini Batch API (50% cost savings).

    Supports both local images and GCS images:
    - Local: carwash gemini-batch input.csv ./images prompt.txt
    - GCS: carwash gemini-batch gs://bucket/path/images prompt.txt
    - GCS with CSV filter: carwash gemini-batch input.csv gs://bucket/path prompt.txt
    """
    console.print(Panel.fit("🚀 Gemini Batch Inference", style="bold magenta"))
    console.print()
    console.print("[cyan]Using Gemini Batch API for 50% cost savings[/cyan]")
    console.print("[cyan]Note: Batch jobs may take up to 24 hours to complete[/cyan]")
    console.print()

    # Interactive input if arguments not provided
    if images_dir is None:
        images_dir_str = Prompt.ask("[cyan]📁 Images path (local dir or GCS: gs://bucket/path)[/cyan]")
        images_dir = images_dir_str
    else:
        images_dir_str = images_dir

    # Check if using GCS path
    is_gcs_path = images_dir_str.startswith("gs://") or (
        "/" in images_dir_str and not images_dir_str.startswith("/") and not Path(images_dir_str).exists()
    )

    # Track temp CSV path if downloaded from GCS
    temp_csv_path: Path | None = None

    if is_gcs_path:
        console.print(f"[cyan]Using GCS images: {images_dir_str}[/cyan]")
        # For GCS, input_csv is optional
        if input_csv is None:
            csv_str = Prompt.ask(
                "[cyan]📄 Input CSV file path (optional, press Enter to use all GCS images)[/cyan]", default=""
            )
            if csv_str and csv_str.strip():
                # Check if CSV is also in GCS
                if csv_str.startswith("gs://") or (
                    "/" in csv_str and not csv_str.startswith("/") and not Path(csv_str).exists()
                ):
                    # Download CSV from GCS
                    import tempfile

                    from src.inference.gemini_batch import download_from_gcs

                    gcs_csv_uri = csv_str if csv_str.startswith("gs://") else f"gs://{csv_str}"
                    temp_csv_path = Path(tempfile.gettempdir()) / "gemini_batch" / "input.csv"
                    temp_csv_path.parent.mkdir(parents=True, exist_ok=True)

                    console.print(f"[cyan]Downloading CSV from GCS: {gcs_csv_uri}[/cyan]")
                    try:
                        download_from_gcs(gcs_csv_uri, temp_csv_path)
                        input_csv = temp_csv_path
                        console.print(f"[green]Downloaded CSV to: {temp_csv_path}[/green]")
                    except Exception as e:
                        error_msg = str(e)
                        if "403" in error_msg or "Forbidden" in error_msg:
                            console.print(f"[red]❌ Permission denied accessing GCS: {gcs_csv_uri}[/red]")
                            console.print(
                                "[yellow]Check that your service account has 'Storage Object Viewer' role.[/yellow]"
                            )
                        else:
                            console.print(f"[red]❌ Error downloading CSV from GCS: {e}[/red]")
                        raise typer.Exit(1) from e
                else:
                    input_csv = _resolve_path(csv_str)
    else:
        # For local, input_csv is required
        if input_csv is None:
            input_csv_str = Prompt.ask("[cyan]📄 Input CSV file path[/cyan]")
            input_csv = _resolve_path(input_csv_str)

    if prompt is None:
        prompt_str = Prompt.ask("[cyan]📝 Prompt file path[/cyan]")
        prompt = _resolve_path(prompt_str)

    # Ask for limit if not provided
    if limit is None:
        limit_str = Prompt.ask("[cyan]🔢 Maximum number of images to process (press Enter for all)[/cyan]", default="")
        if limit_str and limit_str.strip():
            try:
                limit = int(limit_str)
            except ValueError:
                console.print("[yellow]⚠️  Invalid number, processing all images[/yellow]")
                limit = None

    console.print()

    # Validate paths
    if input_csv is not None and not input_csv.exists():
        console.print(f"[red]❌ Error: Input CSV file not found: {input_csv}[/red]")
        raise typer.Exit(1)

    if not is_gcs_path:
        local_images_dir = Path(images_dir_str)
        if not local_images_dir.exists():
            console.print(f"[red]❌ Error: Images directory not found: {local_images_dir}[/red]")
            raise typer.Exit(1)

    if not prompt.exists():
        console.print(f"[red]❌ Error: Prompt file not found: {prompt}[/red]")
        raise typer.Exit(1)

    # Auto-generate output path
    prompt_name = prompt.stem
    if is_gcs_path:
        # Extract dataset name from GCS path
        gcs_parts = images_dir_str.replace("gs://", "").split("/")
        dataset_name = gcs_parts[-1] if gcs_parts[-1] else gcs_parts[-2] if len(gcs_parts) > 1 else "gcs"
    else:
        local_dir = Path(images_dir_str)
        dataset_name = local_dir.parent.name if local_dir.name == "images" else local_dir.name
    model_suffix = model.replace("-", "_") if model else "gemini"
    output = Path("results") / f"{dataset_name}_{prompt_name}_{model_suffix}_batch_result.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_path = output.with_suffix(".log")
    logger.add(
        log_path,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="INFO",
        rotation=None,
        mode="w",
    )

    # Get model from env if not specified
    import os

    from dotenv import load_dotenv

    load_dotenv()
    if model is None:
        model = os.getenv("MODEL_NAME", "gemini-2.5-flash")

    # Check if using Vertex AI (requires GCS bucket)
    use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "FALSE").upper() == "TRUE"

    # Ask for GCS bucket if using Vertex AI and not provided
    if use_vertex and gcs_bucket is None:
        gcs_bucket_str = Prompt.ask(
            "[cyan]🪣 GCS bucket name (for Vertex AI batch jobs)[/cyan]", default=os.getenv("GCS_BUCKET_NAME", "")
        )
        if gcs_bucket_str and gcs_bucket_str.strip():
            gcs_bucket = gcs_bucket_str.strip()

    # Display configuration
    config_table = Table(title="⚙️  Configuration", show_header=False, box=None)
    config_table.add_column("Key", style="cyan", width=20)
    config_table.add_column("Value", style="white")

    config_table.add_row("📄 Input CSV", str(input_csv) if input_csv else "(all GCS images)")
    config_table.add_row("📁 Images", images_dir_str + (" (GCS)" if is_gcs_path else " (local)"))
    config_table.add_row("📝 Prompt file", str(prompt))
    config_table.add_row("💾 Output CSV", str(output))
    config_table.add_row("🤖 Model", model)
    config_table.add_row("🔢 Max Tokens", str(max_tokens))
    config_table.add_row("🌡️  Temperature", str(temperature))
    config_table.add_row("⏱️  Poll Interval", f"{poll_interval}s")
    if gcs_bucket:
        config_table.add_row("🪣 GCS Bucket", gcs_bucket)
    if limit:
        config_table.add_row("📊 Limit", str(limit))

    console.print(config_table)
    console.print()

    # Log configuration
    logger.info("=" * 60)
    logger.info("GEMINI BATCH INFERENCE STARTED")
    logger.info("=" * 60)
    logger.info(f"Input CSV: {input_csv}")
    logger.info(f"Images: {images_dir_str} ({'GCS' if is_gcs_path else 'local'})")
    logger.info(f"Prompt file: {prompt}")
    logger.info(f"Output CSV: {output}")
    logger.info(f"Model: {model}")
    logger.info(f"Max Tokens: {max_tokens}")
    logger.info(f"Poll Interval: {poll_interval}s")
    if gcs_bucket:
        logger.info(f"GCS Bucket: {gcs_bucket}")
    if limit:
        logger.info(f"Limit: {limit}")
    logger.info("-" * 60)

    try:
        summary = run_gemini_batch_inference(
            input_csv=input_csv,
            images_dir=images_dir_str,
            prompt_path=prompt,
            output_csv=output,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            skip_confirmation=yes,
            limit=limit,
            poll_interval=poll_interval,
            gcs_bucket=gcs_bucket,
        )

        # Check if cancelled
        if summary.get("cancelled"):
            raise typer.Exit(0)

        # Display summary
        console.print()
        summary_table = Table(title="📊 Summary", show_header=False, box=None)
        summary_table.add_column("Key", style="cyan", width=30)
        summary_table.add_column("Value", style="white")

        summary_table.add_row("🖼️  Total images", str(summary["total"]))
        summary_table.add_row("✅ Successful", f"[green]{summary['successful']}[/green]")
        summary_table.add_row("❌ Failed", f"[red]{summary['failed']}[/red]")
        summary_table.add_row("💾 Results saved to", str(summary["output_path"]))

        console.print(summary_table)
        console.print()

        # Display timing information
        timing = summary.get("timing", {})
        if timing:
            timing_table = Table(title="⏱️  Timing Information", show_header=False, box=None)
            timing_table.add_column("Key", style="cyan", width=30)
            timing_table.add_column("Value", style="white")

            if timing.get("job_submitted_at"):
                timing_table.add_row("📤 Job Submitted", timing["job_submitted_at"])
            if timing.get("job_started_at"):
                timing_table.add_row("▶️  Job Started", timing["job_started_at"])
            if timing.get("job_completed_at"):
                timing_table.add_row("✅ Job Completed", timing["job_completed_at"])
            if timing.get("total_duration_seconds"):
                duration_str = f"{float(timing['total_duration_seconds']):.2f}s"
                timing_table.add_row("⏱️  Total Duration", f"[bold yellow]{duration_str}[/bold yellow]")
            if timing.get("actual_processing_seconds"):
                processing_str = f"{float(timing['actual_processing_seconds']):.2f}s"
                timing_table.add_row("⚡ Actual Processing", f"[bold cyan]{processing_str}[/bold cyan]")

            console.print(timing_table)
            console.print()

        # Display cost information
        cost = summary.get("cost", {})
        if cost:
            cost_table = Table(title="💰 Cost Information", show_header=False, box=None)
            cost_table.add_column("Key", style="cyan", width=30)
            cost_table.add_column("Value", style="white")

            cost_table.add_row("📥 Input Tokens", str(cost.get("input_tokens", 0)))
            cost_table.add_row("📤 Output Tokens", str(cost.get("output_tokens", 0)))
            cost_table.add_row("📊 Total Tokens", str(cost.get("total_tokens", 0)))
            cost_table.add_row("💵 Input Cost (USD)", cost.get("input_cost_usd", "$0.00"))
            cost_table.add_row("💵 Output Cost (USD)", cost.get("output_cost_usd", "$0.00"))
            cost_table.add_row("💰 Total Cost (USD)", f"[bold green]{cost.get('total_cost_usd', '$0.00')}[/bold green]")
            cost_table.add_row("💴 Total Cost (KRW)", f"[bold green]{cost.get('total_cost_krw', '₩0.00')}[/bold green]")
            cost_table.add_row("🏷️  Batch Pricing", "Yes (50% input discount)" if cost.get("is_batch_pricing") else "No")

            console.print(cost_table)
            console.print()

        # Log summary
        logger.info("=" * 60)
        logger.info("GEMINI BATCH INFERENCE COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Total images: {summary['total']}")
        logger.info(f"Successful: {summary['successful']}")
        logger.info(f"Failed: {summary['failed']}")
        if timing:
            logger.info(f"Total Duration: {timing.get('total_duration_seconds', 'N/A')}s")
            logger.info(f"Actual Processing: {timing.get('actual_processing_seconds', 'N/A')}s")
        if cost:
            logger.info(f"Total Cost (USD): {cost.get('total_cost_usd', 'N/A')}")
            logger.info(f"Total Cost (KRW): {cost.get('total_cost_krw', 'N/A')}")
        logger.info(f"Results saved to: {summary['output_path']}")
        logger.info(f"Log saved to: {log_path}")
        logger.info("=" * 60)

        # Show completion message
        if summary["successful"] == 0 and summary["failed"] > 0:
            console.print(Panel.fit("❌ All inference requests failed!", style="bold red"))
            raise typer.Exit(1)
        elif summary["failed"] > 0:
            success_rate = (summary["successful"] / summary["total"]) * 100
            msg = f"⚠️  Completed with {summary['failed']} failures ({success_rate:.1f}% success rate)"
            console.print(Panel.fit(msg, style="bold yellow"))
        else:
            console.print(Panel.fit("✓ Gemini batch inference completed successfully!", style="bold green"))

    except Exception as e:
        logger.error(f"Gemini batch inference failed: {e}")
        console.print(f"[red]❌ Error: {e}[/red]")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1) from e


@app.command("generate-prompt")
def generate_prompt_cmd() -> None:
    """Generate prompt template from guideline CSV and car parts CSV."""
    console.print(Panel.fit("📝 Prompt Generation", style="bold magenta"))
    console.print()

    # Interactive input for guideline path
    guideline_str = Prompt.ask("[cyan]📄 Guideline CSV file path[/cyan]")
    guideline = _resolve_path(guideline_str)

    if not guideline.exists():
        console.print(f"[red]❌ Error: Guideline file not found: {guideline}[/red]")
        raise typer.Exit(1)

    # Interactive input for car parts path
    car_parts_str = Prompt.ask("[cyan]🚗 Car parts CSV file path[/cyan]")
    car_parts = _resolve_path(car_parts_str) if car_parts_str.strip() else None

    if car_parts and not car_parts.exists():
        console.print(f"[red]❌ Error: Car parts file not found: {car_parts}[/red]")
        raise typer.Exit(1)

    # Interactive input for output path (optional)
    output_str = Prompt.ask("[cyan]💾 Output file path (press Enter for auto)[/cyan]", default="")
    output = _resolve_path(output_str) if output_str.strip() else None

    console.print()

    try:
        # Step 1: Parse guideline and car parts
        console.print("[bold]Step 1:[/bold] Parsing CSV files")
        console.print("-" * 60)
        parsed_rows, part_to_details, exterior_parts, interior_parts, valid_levels = parse_guideline(
            guideline, car_parts
        )
        console.print(f"  Parsed [green]{len(parsed_rows)}[/green] guideline entries")
        console.print(f"  Exterior parts: [green]{len(exterior_parts)}[/green]")
        console.print(f"  Interior parts: [green]{len(interior_parts)}[/green]")
        console.print(f"  Contamination types: [green]{len(valid_levels)}[/green]")

        # Step 2: Generate prompt
        console.print()
        console.print("[bold]Step 2:[/bold] Generating prompt template")
        console.print("-" * 60)
        prompt_text = generate_prompt(parsed_rows, part_to_details, exterior_parts, interior_parts, valid_levels)
        console.print("  [green]✓[/green] Prompt generated successfully")

        # Determine output path based on guideline version
        if output is None:
            guideline_stem = guideline.stem
            version_str = guideline_stem.split("_v")[1] if "_v" in guideline_stem else "5"
            prompts_dir = Path("prompts")
            prompts_dir.mkdir(exist_ok=True)
            output = prompts_dir / f"prompt_v{version_str}.txt"

        # Step 3: Save prompt
        console.print()
        console.print("[bold]Step 3:[/bold] Saving prompt")
        console.print("-" * 60)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            f.write(prompt_text)
        console.print(f"  [green]✓[/green] Saved to: {output}")

        console.print()
        console.print(Panel.fit("✓ Prompt generation completed!", style="bold green"))
        console.print(f"\n[cyan]Generated file:[/cyan] {output}")

    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1) from e


@app.command("resume-job")
def resume_batch_job(
    job_file: Annotated[Path, typer.Argument(help="Path to .job.json file from previous batch run")],
    poll_interval: Annotated[int, typer.Option(help="Seconds between status checks")] = 30,
) -> None:
    """Resume a batch job and get results.

    Use this to retrieve results from a previously started batch job.
    The job file is automatically created when running gemini-batch.
    """
    import json

    from dotenv import load_dotenv

    load_dotenv()

    console.print(Panel.fit("🔄 Resume Batch Job", style="bold magenta"))
    console.print()

    if not job_file.exists():
        console.print(f"[red]❌ Job file not found: {job_file}[/red]")
        raise typer.Exit(1)

    # Load job info
    with open(job_file, encoding="utf-8") as f:
        job_info = json.load(f)

    job_name = job_info["job_name"]
    model = job_info["model"]
    output_csv = Path(job_info["output_csv"])
    prompt_version = job_info["prompt_version"]
    request_keys = job_info["request_keys"]
    use_vertex = job_info.get("use_vertex_ai", True)

    console.print(f"[cyan]Job name: {job_name}[/cyan]")
    console.print(f"[cyan]Model: {model}[/cyan]")
    console.print(f"[cyan]Output: {output_csv}[/cyan]")
    console.print(f"[cyan]Requests: {len(request_keys)}[/cyan]")
    console.print()

    # Setup logging
    log_path = output_csv.with_suffix(".log")
    logger.add(
        log_path,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="INFO",
        rotation=None,
        mode="a",  # Append to existing log
    )

    import os

    from google import genai

    # Initialize client
    if use_vertex:
        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

        if model and model.startswith("gemini-3"):
            location = "global"

        if not project:
            creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if creds_path and Path(creds_path).exists():
                try:
                    with open(creds_path, encoding="utf-8") as f:
                        creds_data = json.load(f)
                        project = creds_data.get("project_id")
                except (OSError, json.JSONDecodeError):
                    pass  # Ignore credential file read errors, will check project below

        if not project:
            console.print("[red]❌ GOOGLE_CLOUD_PROJECT not set[/red]")
            raise typer.Exit(1)

        client = genai.Client(vertexai=True, project=project, location=location)
        console.print(f"[cyan]Using Vertex AI (project: {project})[/cyan]")
    else:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            console.print("[red]❌ GOOGLE_API_KEY not set[/red]")
            raise typer.Exit(1)
        client = genai.Client(api_key=api_key)

    # Import helpers
    from src.inference.gemini_batch import (
        BatchTimingInfo,
        create_csv_row_from_result,
        parse_batch_results,
        poll_batch_job,
    )

    try:
        # Check job status first
        console.print("[cyan]Checking job status...[/cyan]")
        job = client.batches.get(name=job_name)

        state = str(job.state) if hasattr(job, "state") else "UNKNOWN"
        console.print(f"[cyan]Current state: {state}[/cyan]")

        # If job is still running, poll for completion
        if "SUCCEEDED" not in state and "FAILED" not in state and "CANCELLED" not in state:
            console.print("[cyan]Job still running, waiting for completion...[/cyan]")
            final_job, timing_info = poll_batch_job(
                client=client,
                job_name=job_name,
                poll_interval=poll_interval,
            )
        else:
            final_job = job
            _ = BatchTimingInfo()  # Placeholder for future timing info usage

        # Parse results
        console.print("[cyan]Parsing batch results...[/cyan]")
        results, cost_info = parse_batch_results(client, final_job, request_keys, use_vertex_ai=use_vertex)

        # Save results to CSV
        import csv

        output_csv.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "file_name",
            "GT",
            "model",
            "prompt_version",
            "area_name",
            "sub_area",
            "contamination_type",
            "max_severity",
            "success",
            "error",
            "raw_response",
        ]

        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            successful = 0
            failed = 0

            for key, result in results.items():
                row = create_csv_row_from_result(
                    file_name=key,
                    original_data={"file_name": key, "GT": ""},
                    result=result,
                    model_name=model,
                    prompt_version=prompt_version,
                )
                writer.writerow(row)
                if result.get("success"):
                    successful += 1
                else:
                    failed += 1

        console.print()
        console.print(f"[green]✅ Results saved to: {output_csv}[/green]")
        console.print(f"[green]   Successful: {successful}[/green]")
        console.print(f"[red]   Failed: {failed}[/red]")

        # Clean up job file
        job_file.unlink()
        console.print(f"[dim]Removed job file: {job_file}[/dim]")

    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1) from e


@app.command("benchmark-cache")
def benchmark_cache(
    input_csv: Annotated[Path | None, typer.Argument(help="Input CSV file with file_name column")] = None,
    images_dir: Annotated[Path | None, typer.Argument(help="Directory containing images")] = None,
    prompt: Annotated[Path | None, typer.Argument(help="Path to prompt file")] = None,
    model: Annotated[str, typer.Option(help="Model name")] = "qwen3-vl-8b-instruct",
    max_tokens: Annotated[int, typer.Option(help="Maximum tokens to generate")] = 1000,
    temperature: Annotated[float, typer.Option(help="Sampling temperature")] = 0.0,
    limit: Annotated[int | None, typer.Option(help="Number of images to benchmark")] = 1000,
    max_workers: Annotated[int, typer.Option(help="Number of parallel workers for system mode")] = 12,
    api_url: Annotated[str | None, typer.Option(help="VLM API URL")] = None,
    internal: Annotated[bool, typer.Option("--internal", help="Use internal Kubernetes service URL")] = False,
) -> None:
    """Benchmark vLLM prefix caching performance.

    Compares inference speed between:
    - system mode: prompt in system message (enables prefix caching)
    - user mode: prompt in user message (no prefix caching)
    """
    console.print(Panel.fit("⚡ Prefix Caching Benchmark", style="bold magenta"))
    console.print()

    # Interactive input if arguments not provided
    if input_csv is None:
        input_csv_str = Prompt.ask("[cyan]📄 Input CSV file path[/cyan]")
        input_csv = _resolve_path(input_csv_str)

    if images_dir is None:
        images_dir_str = Prompt.ask("[cyan]📁 Images directory path[/cyan]")
        images_dir = _resolve_path(images_dir_str)

    if prompt is None:
        prompt_str = Prompt.ask("[cyan]📝 Prompt file path[/cyan]")
        prompt = _resolve_path(prompt_str)

    # Ask for limit if not provided
    if limit is None:
        limit_str = Prompt.ask("[cyan]🔢 Number of images to benchmark (default: 1000)[/cyan]", default="1000")
        try:
            limit = int(limit_str)
        except ValueError:
            limit = 1000

    # Determine namespace
    namespace = "vllm-test"
    if api_url is None and not internal:
        internal_str = Prompt.ask("[cyan]🔧 Running inside Kubernetes cluster? (y/N)[/cyan]", default="N")
        internal = internal_str.strip().lower() in ["y", "yes"]

    if api_url is None:
        namespace = Prompt.ask(
            "[cyan]🏷️  Select namespace (vllm / vllm-test)[/cyan]", choices=["vllm", "vllm-test"], default="vllm-test"
        )

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

    # Determine API URL
    if api_url is None:
        api_url = _get_api_url(internal, model, namespace)

    # Check server health
    console.print(f"[cyan]🌐 API URL: {api_url}[/cyan]")
    console.print("Checking server health...")
    temp_client = VLMClient(api_url=api_url, model=model)
    health_result = temp_client.check_health(timeout=10)

    if not health_result["healthy"]:
        console.print(f"[red]❌ Server not healthy: {health_result.get('error', 'Unknown error')}[/red]")
        raise typer.Exit(1)

    console.print("[green]✓ Server is healthy[/green]")
    console.print()

    # Display benchmark configuration
    config_table = Table(title="⚙️  Benchmark Configuration", show_header=False, box=None)
    config_table.add_column("Key", style="cyan", width=20)
    config_table.add_column("Value", style="white")

    config_table.add_row("📄 Input CSV", str(input_csv))
    config_table.add_row("📁 Images directory", str(images_dir))
    config_table.add_row("📝 Prompt file", str(prompt))
    config_table.add_row("🌐 API URL", api_url)
    config_table.add_row("🤖 Model", model)
    config_table.add_row("📊 Images to test", str(limit))
    config_table.add_row("⚡ Workers", str(max_workers))

    console.print(config_table)
    console.print()

    # Create output directory for benchmark results
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_dir = Path("results") / f"benchmark_{timestamp}"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[cyan]📁 Benchmark results will be saved to: {benchmark_dir}[/cyan]")
    console.print()

    results = {}
    first_run = True

    for mode in ["system", "user"]:
        # Wait between runs to allow server to stabilize (GPU memory cleanup)
        if not first_run:
            console.print("[dim]Waiting 10s for server to stabilize...[/dim]")
            time.sleep(10)
        first_run = False

        mode_label = "System prompt (prefix caching)" if mode == "system" else "User prompt (no caching)"
        console.print(f"[bold cyan]Running benchmark: {mode_label}[/bold cyan]")
        console.print()

        output_csv = benchmark_dir / f"benchmark_{mode}.csv"

        try:
            summary = run_batch_inference(
                input_csv=input_csv,
                images_dir=images_dir,
                prompt_path=prompt,
                output_csv=output_csv,
                api_url=api_url,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                limit=limit,
                max_workers=max_workers if mode == "system" else 4,
                prompt_mode=mode,
            )

            actual_workers = max_workers if mode == "system" else 4
            results[mode] = {
                "total": summary["total"],
                "successful": summary["successful"],
                "failed": summary["failed"],
                "total_time": summary["total_time"],
                "avg_latency": summary["avg_latency"],
                "avg_time_per_image": summary["avg_time_per_image"],
                "throughput": summary["total"] / summary["total_time"] if summary["total_time"] > 0 else 0,
                "output_csv": output_csv,
                "workers": actual_workers,
            }

            if summary["failed"] > 0:
                console.print(
                    f"[yellow]⚠ {mode_label}: {summary['total_time']:.2f}s "
                    f"({summary['failed']}/{summary['total']} failed)[/yellow]"
                )
                # Show error samples
                import pandas as pd

                df = pd.read_csv(output_csv)
                errors = df[~df["success"]]["error"].dropna().head(3).tolist()
                if errors:
                    console.print("[yellow]  Error samples:[/yellow]")
                    for err in errors:
                        err_str = str(err)[:100]
                        console.print(f"    - {err_str}")
            else:
                console.print(f"[green]✓ {mode_label}: {summary['total_time']:.2f}s[/green]")
            console.print()

        except Exception as e:
            console.print(f"[red]❌ Error in {mode} mode: {e}[/red]")
            results[mode] = {}  # Empty dict instead of None

    # Display comparison results
    console.print()
    console.print(Panel.fit("📊 Benchmark Results", style="bold green"))
    console.print()

    if results.get("system") and results.get("user"):
        system_result = results["system"]
        user_result = results["user"]

        result_table = Table(title="Prefix Caching Comparison", show_header=True)
        result_table.add_column("Metric", style="cyan", width=25)
        result_table.add_column("System (cached)", style="green", justify="right")
        result_table.add_column("User (no cache)", style="yellow", justify="right")
        result_table.add_column("Difference", style="magenta", justify="right")

        # Total time comparison
        time_diff = user_result["total_time"] - system_result["total_time"]
        time_pct = (time_diff / user_result["total_time"]) * 100 if user_result["total_time"] > 0 else 0
        result_table.add_row(
            "Total time",
            f"{system_result['total_time']:.2f}s",
            f"{user_result['total_time']:.2f}s",
            f"{time_diff:+.2f}s ({time_pct:+.1f}%)",
        )

        # Throughput comparison
        throughput_diff = system_result["throughput"] - user_result["throughput"]
        throughput_pct = (throughput_diff / user_result["throughput"]) * 100 if user_result["throughput"] > 0 else 0
        result_table.add_row(
            "Throughput",
            f"{system_result['throughput']:.2f} img/s",
            f"{user_result['throughput']:.2f} img/s",
            f"{throughput_diff:+.2f} ({throughput_pct:+.1f}%)",
        )

        # Avg latency comparison
        latency_diff = user_result["avg_latency"] - system_result["avg_latency"]
        latency_pct = (latency_diff / user_result["avg_latency"]) * 100 if user_result["avg_latency"] > 0 else 0
        result_table.add_row(
            "Avg API latency",
            f"{system_result['avg_latency']:.3f}s",
            f"{user_result['avg_latency']:.3f}s",
            f"{latency_diff:+.3f}s ({latency_pct:+.1f}%)",
        )

        # Avg time per image
        time_per_img_diff = user_result["avg_time_per_image"] - system_result["avg_time_per_image"]
        time_per_img_pct = (
            (time_per_img_diff / user_result["avg_time_per_image"]) * 100
            if user_result["avg_time_per_image"] > 0
            else 0
        )
        result_table.add_row(
            "Avg time per image",
            f"{system_result['avg_time_per_image']:.3f}s",
            f"{user_result['avg_time_per_image']:.3f}s",
            f"{time_per_img_diff:+.3f}s ({time_per_img_pct:+.1f}%)",
        )

        # Success rate
        system_success_rate = (
            (system_result["successful"] / system_result["total"]) * 100 if system_result["total"] > 0 else 0
        )
        user_success_rate = (user_result["successful"] / user_result["total"]) * 100 if user_result["total"] > 0 else 0
        result_table.add_row(
            "Success rate",
            f"{system_success_rate:.1f}%",
            f"{user_success_rate:.1f}%",
            f"{system_success_rate - user_success_rate:+.1f}%",
        )

        # Workers
        result_table.add_row(
            "Workers",
            f"{system_result['workers']}",
            f"{user_result['workers']}",
            "-",
        )

        console.print(result_table)
        console.print()

        # Summary
        speedup = user_result["total_time"] / system_result["total_time"] if system_result["total_time"] > 0 else 1

        if speedup > 1.05:
            console.print(f"[bold green]✓ Prefix caching provides {speedup:.2f}x speedup![/bold green]")
        elif speedup < 0.95:
            console.print(
                f"[bold yellow]⚠ User mode was faster ({1 / speedup:.2f}x). "
                "Prefix caching may not be effective for this workload.[/bold yellow]"
            )
        else:
            console.print("[bold cyan]≈ Performance is similar between both modes.[/bold cyan]")

    else:
        console.print("[red]Could not complete benchmark comparison.[/red]")
        if results.get("system"):
            console.print(f"  System mode: {results['system']['total_time']:.2f}s")
        if results.get("user"):
            console.print(f"  User mode: {results['user']['total_time']:.2f}s")

    # Estimate time for 140,000 images
    console.print()
    console.print(Panel.fit("📈 140,000 Images Processing Time Estimate", style="bold blue"))
    console.print()

    target_images = 140000

    estimate_table = Table(title="Estimated Processing Time for 140K Images", show_header=True)
    estimate_table.add_column("Mode", style="cyan", width=25)
    estimate_table.add_column("Throughput", style="white", justify="right")
    estimate_table.add_column("Estimated Time", style="yellow", justify="right")
    estimate_table.add_column("Estimated Hours", style="green", justify="right")

    if results.get("system") and results["system"].get("throughput", 0) > 0:
        system_throughput = results["system"]["throughput"]
        system_est_seconds = target_images / system_throughput
        system_est_hours = system_est_seconds / 3600
        estimate_table.add_row(
            "System (prefix caching)",
            f"{system_throughput:.2f} img/s",
            f"{system_est_seconds:.0f}s",
            f"[bold green]{system_est_hours:.1f}h[/bold green]",
        )
    else:
        estimate_table.add_row("System (prefix caching)", "N/A", "N/A", "N/A")

    if results.get("user") and results["user"].get("throughput", 0) > 0:
        user_throughput = results["user"]["throughput"]
        user_est_seconds = target_images / user_throughput
        user_est_hours = user_est_seconds / 3600
        estimate_table.add_row(
            "User (no caching)",
            f"{user_throughput:.2f} img/s",
            f"{user_est_seconds:.0f}s",
            f"[bold yellow]{user_est_hours:.1f}h[/bold yellow]",
        )
    else:
        estimate_table.add_row("User (no caching)", "N/A", "N/A", "N/A")

    console.print(estimate_table)

    # Show time savings
    if (
        results.get("system")
        and results.get("user")
        and results["system"].get("throughput", 0) > 0
        and results["user"].get("throughput", 0) > 0
    ):
        system_est_hours = target_images / results["system"]["throughput"] / 3600
        user_est_hours = target_images / results["user"]["throughput"] / 3600
        time_saved_hours = user_est_hours - system_est_hours

        if time_saved_hours > 0:
            console.print()
            console.print(
                f"[bold green]💡 Prefix caching saves approximately {time_saved_hours:.1f} hours "
                f"for 140K images![/bold green]"
            )

    # Run comparison analysis if both results exist
    if results.get("system") and results.get("user"):
        system_csv = results["system"].get("output_csv")
        user_csv = results["user"].get("output_csv")

        if system_csv and user_csv and Path(system_csv).exists() and Path(user_csv).exists():
            console.print()
            console.print(Panel.fit("🔍 Response Comparison Analysis", style="bold cyan"))
            console.print()

            from src.inference.compare import compare_benchmark_results

            comparison = compare_benchmark_results(Path(system_csv), Path(user_csv))

            # Save comparison report
            comparison_file = benchmark_dir / "comparison_report.json"
            import json

            with open(comparison_file, "w", encoding="utf-8") as f:
                json.dump(comparison, f, indent=2, ensure_ascii=False)

            # Display comparison metrics
            comp_table = Table(title="Response Comparison", show_header=False, box=None)
            comp_table.add_column("Metric", style="cyan", width=30)
            comp_table.add_column("Value", style="white")

            comp_table.add_row("📊 Total compared", str(comparison["total_compared"]))
            comp_table.add_row(
                "✅ Identical responses",
                f"[green]{comparison['identical']} ({comparison['identical_pct']:.1f}%)[/green]",
            )
            comp_table.add_row(
                "⚠️  Different responses",
                f"[yellow]{comparison['different']} ({comparison['different_pct']:.1f}%)[/yellow]",
            )
            comp_table.add_row("❌ Parse errors (system)", str(comparison["system_parse_errors"]))
            comp_table.add_row("❌ Parse errors (user)", str(comparison["user_parse_errors"]))

            console.print(comp_table)
            console.print()
            console.print(f"[dim]Detailed comparison saved to: {comparison_file}[/dim]")

    console.print()
    console.print(f"[bold green]✓ Benchmark results saved to: {benchmark_dir}[/bold green]")
    console.print()


def main() -> None:
    """Entry point for CLI."""
    # Show help if no arguments provided
    if len(sys.argv) == 1:
        sys.argv.append("--help")

    app()


if __name__ == "__main__":
    main()
