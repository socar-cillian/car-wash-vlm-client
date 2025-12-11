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
from src.inference import parse_batch_output, run_batch_inference, run_gemini_batch_inference
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
def batch_inference_menu() -> None:
    """Run batch inference on images - Interactive menu."""
    from rich.box import ROUNDED

    # Header
    header_content = "[bold white]🚗 Batch Inference[/bold white]\n[dim]vLLM 서버를 사용한 배치 추론[/dim]"
    console.print(Panel(header_content, style="magenta", box=ROUNDED))
    console.print()

    # Menu options
    menu_table = Table(
        show_header=True,
        header_style="bold cyan",
        box=ROUNDED,
        border_style="cyan",
        padding=(0, 1),
    )
    menu_table.add_column("번호", justify="center", width=6)
    menu_table.add_column("모드", width=20)
    menu_table.add_column("설명", style="dim")

    menu_table.add_row("[bold green]1[/]", "📄 CSV 모드", "CSV 파일 기반 추론 (GT 비교용)")
    menu_table.add_row("[bold green]2[/]", "📁 폴더 모드", "폴더 내 모든 이미지 추론 (raw response 수집)")
    menu_table.add_row("[bold red]0[/]", "🚪 종료", "종료")

    console.print(menu_table)
    console.print()

    # Get user choice
    choice = Prompt.ask(
        "[bold cyan]선택[/bold cyan]",
        choices=["0", "1", "2"],
        default="1",
    )

    console.print()

    if choice == "0":
        console.print("[yellow]Exiting...[/yellow]")
        return
    elif choice == "1":
        _batch_infer_csv_mode()
    elif choice == "2":
        _batch_infer_folder_mode()


def _batch_infer_csv_mode() -> None:
    """Run batch inference with CSV input (for GT comparison)."""
    console.print(Panel.fit("📄 CSV 모드 - GT 비교용 배치 추론", style="bold cyan"))
    console.print()

    # Initialize variables
    model = "qwen3-vl-8b-instruct"
    max_tokens: int | None = None
    temperature = 0.0
    limit: int | None = None
    max_workers: int | None = None
    api_url: str | None = None
    internal = False

    # Interactive input
    input_csv_str = Prompt.ask("[cyan]📄 Input CSV file path[/cyan]")
    input_csv = _resolve_path(input_csv_str)

    images_dir_str = Prompt.ask("[cyan]📁 Images directory path[/cyan]")
    images_dir = _resolve_path(images_dir_str)

    prompt_str = Prompt.ask("[cyan]📝 Prompt file path[/cyan]")
    prompt = _resolve_path(prompt_str)

    # Ask for limit
    limit_str = Prompt.ask("[cyan]🔢 Maximum number of images to process (press Enter for all)[/cyan]", default="")
    if limit_str and limit_str.strip():
        try:
            limit = int(limit_str)
        except ValueError:
            console.print("[yellow]⚠️  Invalid number, processing all images[/yellow]")
            limit = None

    # Ask for internal mode
    namespace = "vllm-test"  # default
    internal_str = Prompt.ask("[cyan]🔧 Running inside Kubernetes cluster? (y/N)[/cyan]", default="N")
    internal = internal_str.strip().lower() in ["y", "yes"]

    # Ask for namespace
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

    # Set workers - default 12 for prefix caching optimization
    max_workers = 12  # Optimized for prefix caching with A100 40GB
    console.print(f"[cyan]ℹ️  Using {max_workers} workers (prefix caching enabled)[/cyan]")
    console.print()
    logger.info(f"Using {max_workers} workers")

    # If max_tokens not specified, use a reasonable default (not max_model_len!)
    # max_model_len is total context (input + output), so we can't use it all for output
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

    except typer.Exit:
        raise
    except Exception as e:
        logger.error(f"Batch inference failed with error: {e}")
        console.print(f"[red]❌ Error: {e}[/red]")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1) from e


def _batch_infer_folder_mode() -> None:
    """Run batch inference on all images in a folder (raw response collection)."""
    import json
    from concurrent.futures import ThreadPoolExecutor

    from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

    from src.inference.batch import load_prompt, process_image

    console.print(Panel.fit("📁 폴더 모드 - Raw Response 수집", style="bold cyan"))
    console.print()

    # Initialize variables
    model = "qwen3-vl-8b-instruct"
    max_tokens = 1000
    temperature = 0.0
    limit: int | None = None
    max_workers = 12
    api_url: str | None = None
    internal = False

    # Interactive input
    images_dir_str = Prompt.ask("[cyan]📁 Images directory path[/cyan]")
    images_dir = _resolve_path(images_dir_str)

    prompt_str = Prompt.ask("[cyan]📝 Prompt file path[/cyan]")
    prompt = _resolve_path(prompt_str)

    # Ask for limit
    limit_str = Prompt.ask("[cyan]🔢 Maximum number of images to process (press Enter for all)[/cyan]", default="")
    if limit_str and limit_str.strip():
        try:
            limit = int(limit_str)
        except ValueError:
            console.print("[yellow]⚠️  Invalid number, processing all images[/yellow]")
            limit = None

    # Ask for internal mode
    namespace = "vllm-test"  # default
    internal_str = Prompt.ask("[cyan]🔧 Running inside Kubernetes cluster? (y/N)[/cyan]", default="N")
    internal = internal_str.strip().lower() in ["y", "yes"]

    # Ask for namespace
    namespace = Prompt.ask(
        "[cyan]🏷️  Select namespace (vllm / vllm-test)[/cyan]", choices=["vllm", "vllm-test"], default="vllm-test"
    )

    console.print()

    # Validate paths
    if not images_dir.exists():
        console.print(f"[red]❌ Error: Images directory not found: {images_dir}[/red]")
        raise typer.Exit(1)

    if not prompt.exists():
        console.print(f"[red]❌ Error: Prompt file not found: {prompt}[/red]")
        raise typer.Exit(1)

    # Find all image files in the directory
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    image_files = [f for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
    image_files.sort()

    if not image_files:
        console.print(f"[red]❌ Error: No image files found in {images_dir}[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Found {len(image_files)} image files[/green]")

    # Apply limit if specified
    if limit is not None and limit > 0:
        image_files = image_files[:limit]
        console.print(f"[cyan]Processing first {len(image_files)} images (limit applied)[/cyan]")

    # Auto-generate output path
    prompt_name = prompt.stem
    dataset_name = images_dir.parent.name if images_dir.name == "images" else images_dir.name
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json = Path("results") / f"{dataset_name}_{prompt_name}_{timestamp}_raw.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_path = output_json.with_suffix(".log")
    logger.add(
        log_path,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="INFO",
        rotation=None,
        mode="w",
    )
    logger.info(f"Logging to {log_path}")

    # Determine API URL
    api_url = _get_api_url(internal, model, namespace)

    # Check server health
    console.print(f"[cyan]🌐 API URL: {api_url}[/cyan]")
    console.print("Checking server health...")
    temp_client = VLMClient(api_url=api_url, model=model)
    health_result = temp_client.check_health(timeout=10)

    if not health_result["healthy"]:
        console.print("[yellow]⏳ Server not ready (may be scaled to zero). Waiting for cold start...[/yellow]")
        console.print("[yellow]   This can take 5-10 minutes for GPU pod startup + model loading[/yellow]")
        logger.info("Server not ready, waiting for cold start...")

        max_wait = 900
        wait_interval = 30
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

    console.print()

    # Display configuration
    config_table = Table(title="⚙️  Configuration", show_header=False, box=None)
    config_table.add_column("Key", style="cyan", width=20)
    config_table.add_column("Value", style="white")

    config_table.add_row("📁 Images directory", str(images_dir))
    config_table.add_row("📝 Prompt file", str(prompt))
    config_table.add_row("💾 Output JSON", str(output_json))
    config_table.add_row("🌐 API URL", api_url)
    config_table.add_row("🤖 Model", model)
    config_table.add_row("🔢 Max Tokens", str(max_tokens))
    config_table.add_row("⚡ Workers", str(max_workers))
    config_table.add_row("📊 Images to process", str(len(image_files)))

    console.print(config_table)
    console.print()

    # Load prompt
    prompt_text = load_prompt(prompt)
    prompt_version = prompt.stem

    # Initialize client
    client = VLMClient(api_url=api_url, model=model)

    # Process images
    results: list[dict] = []
    successful_count = 0
    failed_count = 0
    total_start_time = time.time()

    try:
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TextColumn("•"),
            TimeRemainingColumn(),
            refresh_per_second=2,
        ) as progress:
            process_task = progress.add_task("Processing", total=len(image_files))

            with ThreadPoolExecutor(max_workers=max_workers) as executor:

                def process_single(image_path: Path) -> dict:
                    inference_result = process_image(
                        client,
                        str(image_path),
                        prompt_text,
                        max_tokens,
                        temperature,
                        image_path.name,
                        "system",
                    )
                    return {
                        "image_name": image_path.name,
                        "success": inference_result["success"],
                        "latency": inference_result.get("latency", 0),
                        "result": inference_result.get("result"),
                        "error": inference_result.get("error"),
                        "raw_response": inference_result.get("raw_response"),
                    }

                for result in executor.map(process_single, image_files):
                    results.append(result)
                    if result["success"]:
                        successful_count += 1
                    else:
                        failed_count += 1
                    progress.update(process_task, advance=1)

        total_elapsed_time = time.time() - total_start_time

        # Save results to JSON
        output_data = {
            "metadata": {
                "model": model,
                "prompt_version": prompt_version,
                "images_dir": str(images_dir),
                "total_images": len(image_files),
                "successful": successful_count,
                "failed": failed_count,
                "total_time_seconds": total_elapsed_time,
                "timestamp": datetime.now().isoformat(),
            },
            "results": results,
        }

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        # Display summary
        console.print()
        summary_table = Table(title="📊 Summary", show_header=False, box=None)
        summary_table.add_column("Key", style="cyan", width=25)
        summary_table.add_column("Value", style="white")

        throughput = len(image_files) / total_elapsed_time if total_elapsed_time > 0 else 0

        summary_table.add_row("🖼️  Total images", str(len(image_files)))
        summary_table.add_row("✅ Successful", f"[green]{successful_count}[/green]")
        summary_table.add_row("❌ Failed", f"[red]{failed_count}[/red]")
        summary_table.add_row("⏱️  Total time", f"[bold yellow]{total_elapsed_time:.2f}s[/bold yellow]")
        summary_table.add_row("🚀 Throughput", f"[bold magenta]{throughput:.2f} images/sec[/bold magenta]")
        summary_table.add_row("💾 Results saved to", str(output_json))

        console.print(summary_table)
        console.print()

        if failed_count == 0:
            console.print(Panel.fit("✓ Batch inference completed successfully!", style="bold green"))
        elif successful_count == 0:
            console.print(Panel.fit("❌ All inference requests failed!", style="bold red"))
            raise typer.Exit(1)
        else:
            success_rate = (successful_count / len(image_files)) * 100
            msg = f"⚠️  Completed with {failed_count} failures ({success_rate:.1f}% success rate)"
            console.print(Panel.fit(msg, style="bold yellow"))

    except typer.Exit:
        raise
    except Exception as e:
        logger.error(f"Batch inference failed with error: {e}")
        console.print(f"[red]❌ Error: {e}[/red]")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1) from e


def _gemini_batch_infer() -> None:
    """Run batch inference using Gemini Batch API (50% cost savings)."""
    console.print(Panel.fit("🚀 Gemini Batch Inference", style="bold magenta"))
    console.print()
    console.print("[cyan]Using Gemini Batch API for 50% cost savings[/cyan]")
    console.print("[cyan]Note: Batch jobs may take up to 24 hours to complete[/cyan]")
    console.print()

    # Initialize parameters
    input_csv: Path | None = None
    images_dir: str | None = None
    prompt: Path | None = None
    model: str | None = None
    max_tokens: int = 1000
    temperature: float = 0.0
    limit: int | None = None
    poll_interval: int = 30
    yes: bool = False
    gcs_bucket: str | None = None

    # Interactive input
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
    # Add timestamp to output filename
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path("results") / f"{dataset_name}_{prompt_name}_{model_suffix}_{timestamp}.csv"
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

        # If job was just submitted (not waiting for completion), exit early
        if summary.get("status") == "submitted":
            raise typer.Exit(0)

        # Display summary (only when job is completed)
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

    except typer.Exit:
        # Re-raise typer.Exit to allow clean exit
        raise
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


def _gemini_batch_resume() -> None:
    """Resume a batch job and get results."""
    import json

    from dotenv import load_dotenv

    load_dotenv()

    console.print(Panel.fit("🔄 Resume Batch Job", style="bold magenta"))
    console.print()

    # Interactive input
    job_file_str = Prompt.ask("[cyan]📄 Path to .job.json file[/cyan]")
    job_file = _resolve_path(job_file_str)
    poll_interval = 30

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

        # Mark job as completed (rename to .done.json instead of deleting)
        done_file = job_file.with_suffix(".done.json")
        job_file.rename(done_file)
        console.print(f"[dim]Job file moved to: {done_file}[/dim]")

    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1) from e


# =============================================================================
# Gemini Batch Internal Functions: parse, status, info, list
# =============================================================================


def _gemini_batch_parse() -> None:
    """Parse Gemini Batch API output JSONL file."""
    console.print(Panel.fit("📄 Parse Batch Output", style="bold magenta"))
    console.print()

    # Interactive input
    input_str = Prompt.ask("[cyan]📥 Input JSONL file path[/cyan]")
    input_jsonl = _resolve_path(input_str)
    output_jsonl: Path | None = None

    if not input_jsonl.exists():
        console.print(f"[red]❌ Input file not found: {input_jsonl}[/red]")
        raise typer.Exit(1)

    # Ask for output path interactively
    if output_jsonl is None:
        output_str = Prompt.ask(
            "[cyan]📤 Output JSONL path (press Enter to print to stdout)[/cyan]",
            default="",
        )
        if output_str.strip():
            output_jsonl = Path(output_str)

    console.print()
    console.print(f"[cyan]Parsing: {input_jsonl}[/cyan]")

    try:
        results, summary = parse_batch_output(input_jsonl, output_jsonl, verbose=False)

        # Display summary
        summary_table = Table(title="📊 Parse Summary", show_header=False, box=None)
        summary_table.add_column("Key", style="cyan", width=25)
        summary_table.add_column("Value", style="white")

        summary_table.add_row("📄 Total records", str(summary.total_count))
        summary_table.add_row("✅ Successful", f"[green]{summary.success_count}[/green]")
        summary_table.add_row("❌ Failed", f"[red]{summary.failed_count}[/red]")
        summary_table.add_row(
            "📦 Input size",
            f"{summary.input_size_bytes / 1024 / 1024:.1f} MB",
        )
        if summary.output_size_bytes is not None:
            summary_table.add_row(
                "📦 Output size",
                f"{summary.output_size_bytes / 1024 / 1024:.1f} MB",
            )
            if summary.reduction_percent:
                summary_table.add_row(
                    "📉 Size reduction",
                    f"[bold green]{summary.reduction_percent:.1f}%[/bold green]",
                )

        console.print(summary_table)
        console.print()

        if output_jsonl:
            console.print(f"[green]✅ Parsed output saved to: {output_jsonl}[/green]")
        else:
            # Print results to stdout
            console.print("[dim]Results (JSON):[/dim]")
            import json

            for r in results:
                print(
                    json.dumps(
                        {
                            "key": r.key,
                            "success": r.success,
                            "finish_reason": r.finish_reason,
                            "result": r.result,
                            "error": r.error,
                            "input_tokens": r.input_tokens,
                            "output_tokens": r.output_tokens,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )

    except Exception as e:
        console.print(f"[red]❌ Error parsing batch output: {e}[/red]")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1) from e


def _gemini_batch_status() -> None:
    """Check the status of a Gemini batch job."""
    import json
    import os

    from dotenv import load_dotenv
    from google import genai

    load_dotenv()

    console.print(Panel.fit("📊 Batch Job Status", style="bold magenta"))
    console.print()

    # Interactive input
    input_type = Prompt.ask(
        "[cyan]Enter job name or .job.json file path?[/cyan]",
        choices=["name", "file"],
        default="file",
    )

    job_name: str | None = None
    job_file: Path | None = None

    if input_type == "file":
        job_file_str = Prompt.ask("[cyan]📄 Path to .job.json file[/cyan]")
        job_file = _resolve_path(job_file_str)
        if not job_file.exists():
            console.print(f"[red]❌ Job file not found: {job_file}[/red]")
            raise typer.Exit(1)
        with open(job_file, encoding="utf-8") as f:
            job_info = json.load(f)
        job_name = job_info.get("job_name")
        console.print(f"[cyan]Loaded job from: {job_file}[/cyan]")
    else:
        job_name = Prompt.ask("[cyan]🔑 Batch job name[/cyan]")

    if not job_name:
        console.print("[red]❌ Job name is required[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Checking status for: {job_name}[/cyan]")
    console.print()

    # Initialize client
    use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "FALSE").upper() == "TRUE"

    if use_vertex:
        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

        if not project:
            creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if creds_path and Path(creds_path).exists():
                try:
                    with open(creds_path, encoding="utf-8") as f:
                        creds_data = json.load(f)
                        project = creds_data.get("project_id")
                except (OSError, json.JSONDecodeError):
                    pass

        if not project:
            console.print("[red]❌ GOOGLE_CLOUD_PROJECT not set[/red]")
            raise typer.Exit(1)

        client = genai.Client(vertexai=True, project=project, location=location)
    else:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            console.print("[red]❌ GOOGLE_API_KEY not set[/red]")
            raise typer.Exit(1)
        client = genai.Client(api_key=api_key)

    try:
        job = client.batches.get(name=job_name)

        state = str(job.state) if hasattr(job, "state") else "UNKNOWN"

        # State color mapping
        state_colors = {
            "JOB_STATE_SUCCEEDED": "green",
            "JOB_STATE_FAILED": "red",
            "JOB_STATE_CANCELLED": "yellow",
            "JOB_STATE_RUNNING": "cyan",
            "JOB_STATE_PENDING": "yellow",
        }
        state_color = state_colors.get(state, "white")

        from datetime import datetime, timedelta, timezone

        kst_tz = timezone(timedelta(hours=9))

        def format_time_with_kst(dt: datetime | None) -> str:
            """Format datetime with both UTC and KST."""
            if dt is None:
                return "N/A"
            kst_time = dt.astimezone(kst_tz)
            return f"{dt.strftime('%Y-%m-%d %H:%M:%S')} UTC | {kst_time.strftime('%m/%d %H:%M')} KST"

        status_table = Table(title="Job Status", show_header=False, box=None)
        status_table.add_column("Key", style="cyan", width=20)
        status_table.add_column("Value", style="white")

        status_table.add_row("🔑 Job Name", job_name)
        status_table.add_row("📊 State", f"[{state_color}]{state}[/{state_color}]")

        if hasattr(job, "model") and job.model:
            status_table.add_row("🤖 Model", str(job.model))

        if hasattr(job, "create_time") and job.create_time:
            status_table.add_row("📅 Created", format_time_with_kst(job.create_time))

        if hasattr(job, "start_time") and job.start_time:
            status_table.add_row("▶️  Started", format_time_with_kst(job.start_time))

        if hasattr(job, "end_time") and job.end_time:
            status_table.add_row("⏹️  Ended", format_time_with_kst(job.end_time))

        if hasattr(job, "update_time") and job.update_time:
            status_table.add_row("🔄 Updated", format_time_with_kst(job.update_time))

        console.print(status_table)
        console.print()

        # Show completion status message
        if state == "JOB_STATE_SUCCEEDED":
            console.print("[bold green]✅ Job completed successfully![/bold green]")
            if job_file:
                console.print(f"[cyan]Run 'carwash gemini-batch resume {job_file}' to get results[/cyan]")
        elif state == "JOB_STATE_FAILED":
            console.print("[bold red]❌ Job failed[/bold red]")
        elif state == "JOB_STATE_CANCELLED":
            console.print("[bold yellow]⚠️ Job was cancelled[/bold yellow]")
        elif state == "JOB_STATE_RUNNING":
            console.print("[bold cyan]⏳ Job is still running...[/bold cyan]")
        elif state == "JOB_STATE_PENDING":
            console.print("[bold yellow]⏳ Job is queued and waiting to start...[/bold yellow]")

    except Exception as e:
        console.print(f"[red]❌ Error checking job status: {e}[/red]")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1) from e


def _gemini_batch_info() -> None:
    """Get detailed information about a Gemini batch job."""
    import json
    import os

    from dotenv import load_dotenv
    from google import genai

    load_dotenv()

    console.print(Panel.fit("ℹ️  Batch Job Info", style="bold magenta"))
    console.print()

    # Interactive input
    input_type = Prompt.ask(
        "[cyan]Enter job name or .job.json file path?[/cyan]",
        choices=["name", "file"],
        default="file",
    )

    job_name: str | None = None
    local_job_info = None

    if input_type == "file":
        job_file_str = Prompt.ask("[cyan]📄 Path to .job.json file[/cyan]")
        job_file = _resolve_path(job_file_str)
        if not job_file.exists():
            console.print(f"[red]❌ Job file not found: {job_file}[/red]")
            raise typer.Exit(1)
        with open(job_file, encoding="utf-8") as f:
            local_job_info = json.load(f)
        job_name = local_job_info.get("job_name")
        console.print(f"[cyan]Loaded job info from: {job_file}[/cyan]")
    else:
        job_name = Prompt.ask("[cyan]🔑 Batch job name[/cyan]")

    if not job_name:
        console.print("[red]❌ Job name is required[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Fetching details for: {job_name}[/cyan]")
    console.print()

    # Initialize client
    use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "FALSE").upper() == "TRUE"

    if use_vertex:
        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

        if not project:
            creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if creds_path and Path(creds_path).exists():
                try:
                    with open(creds_path, encoding="utf-8") as f:
                        creds_data = json.load(f)
                        project = creds_data.get("project_id")
                except (OSError, json.JSONDecodeError):
                    pass

        if not project:
            console.print("[red]❌ GOOGLE_CLOUD_PROJECT not set[/red]")
            raise typer.Exit(1)

        client = genai.Client(vertexai=True, project=project, location=location)
    else:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            console.print("[red]❌ GOOGLE_API_KEY not set[/red]")
            raise typer.Exit(1)
        client = genai.Client(api_key=api_key)

    try:
        job = client.batches.get(name=job_name)

        # Basic info table
        basic_table = Table(title="📋 Basic Information", show_header=False, box=None)
        basic_table.add_column("Key", style="cyan", width=25)
        basic_table.add_column("Value", style="white")

        basic_table.add_row("🔑 Job Name", job_name)

        state = str(job.state) if hasattr(job, "state") else "UNKNOWN"
        state_colors = {
            "JOB_STATE_SUCCEEDED": "green",
            "JOB_STATE_FAILED": "red",
            "JOB_STATE_CANCELLED": "yellow",
            "JOB_STATE_RUNNING": "cyan",
            "JOB_STATE_PENDING": "yellow",
        }
        state_color = state_colors.get(state, "white")
        basic_table.add_row("📊 State", f"[{state_color}]{state}[/{state_color}]")

        if hasattr(job, "model") and job.model:
            basic_table.add_row("🤖 Model", str(job.model))

        if hasattr(job, "display_name") and job.display_name:
            basic_table.add_row("📛 Display Name", job.display_name)

        console.print(basic_table)
        console.print()

        # Timing table
        timing_table = Table(title="⏱️  Timing", show_header=False, box=None)
        timing_table.add_column("Key", style="cyan", width=25)
        timing_table.add_column("Value", style="white")

        if hasattr(job, "create_time") and job.create_time:
            timing_table.add_row("📅 Created", str(job.create_time))

        if hasattr(job, "start_time") and job.start_time:
            timing_table.add_row("▶️  Started", str(job.start_time))

        if hasattr(job, "end_time") and job.end_time:
            timing_table.add_row("⏹️  Ended", str(job.end_time))

        if hasattr(job, "update_time") and job.update_time:
            timing_table.add_row("🔄 Updated", str(job.update_time))

        # Calculate durations
        if hasattr(job, "start_time") and hasattr(job, "end_time") and job.start_time and job.end_time:
            duration = (job.end_time - job.start_time).total_seconds()
            timing_table.add_row("⏱️  Processing Duration", f"[bold yellow]{duration:.1f}s[/bold yellow]")

        if hasattr(job, "create_time") and hasattr(job, "start_time") and job.create_time and job.start_time:
            queue_wait = (job.start_time - job.create_time).total_seconds()
            timing_table.add_row("⏳ Queue Wait", f"{queue_wait:.1f}s")

        console.print(timing_table)
        console.print()

        # Source/Destination table
        if hasattr(job, "src") or hasattr(job, "dest"):
            io_table = Table(title="📁 Input/Output", show_header=False, box=None)
            io_table.add_column("Key", style="cyan", width=25)
            io_table.add_column("Value", style="white")

            if hasattr(job, "src") and job.src:
                src = job.src
                if hasattr(src, "gcs_uri") and src.gcs_uri:
                    # gcs_uri can be a string or list
                    gcs_uri_val = src.gcs_uri
                    if isinstance(gcs_uri_val, list):
                        gcs_uri_str = ", ".join(str(u) for u in gcs_uri_val)
                    else:
                        gcs_uri_str = str(gcs_uri_val)
                    io_table.add_row("📥 Input GCS URI", gcs_uri_str)
                elif hasattr(src, "inlined_requests") and src.inlined_requests:
                    io_table.add_row("📥 Input Type", f"Inline ({len(src.inlined_requests)} requests)")

            if hasattr(job, "dest") and job.dest:
                dest = job.dest
                if hasattr(dest, "gcs_uri") and dest.gcs_uri:
                    # gcs_uri can be a string or list
                    dest_gcs_uri = dest.gcs_uri
                    if isinstance(dest_gcs_uri, list):
                        gcs_uri_str = ", ".join(str(u) for u in dest_gcs_uri)
                    else:
                        gcs_uri_str = str(dest_gcs_uri)
                    io_table.add_row("📤 Output GCS URI", gcs_uri_str)
                elif hasattr(dest, "inlined_responses") and dest.inlined_responses:
                    io_table.add_row("📤 Output Type", f"Inline ({len(dest.inlined_responses)} responses)")

            console.print(io_table)
            console.print()

        # Local job info if available
        if local_job_info:
            local_table = Table(title="📝 Local Job Info", show_header=False, box=None)
            local_table.add_column("Key", style="cyan", width=25)
            local_table.add_column("Value", style="white")

            if "num_requests" in local_job_info:
                local_table.add_row("📊 Num Requests", str(local_job_info["num_requests"]))

            if "prompt_version" in local_job_info:
                local_table.add_row("📝 Prompt Version", local_job_info["prompt_version"])

            if "output_csv" in local_job_info:
                local_table.add_row("💾 Output CSV", local_job_info["output_csv"])

            if "timing" in local_job_info:
                timing = local_job_info["timing"]
                if timing.get("cli_started_at"):
                    local_table.add_row("🚀 CLI Started", timing["cli_started_at"])
                if timing.get("job_submitted_at"):
                    local_table.add_row("📤 Submitted", timing["job_submitted_at"])

            console.print(local_table)
            console.print()

        # Raw attributes for debugging
        console.print("[dim]Available job attributes:[/dim]")
        attrs = [attr for attr in dir(job) if not attr.startswith("_")]
        console.print(f"[dim]{', '.join(attrs[:20])}{'...' if len(attrs) > 20 else ''}[/dim]")

    except Exception as e:
        console.print(f"[red]❌ Error fetching job info: {e}[/red]")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1) from e


def _gemini_batch_list() -> None:
    """List recent Gemini batch jobs."""
    import glob
    import json
    import os
    from datetime import timedelta, timezone

    from dotenv import load_dotenv
    from google import genai

    load_dotenv()

    console.print(Panel.fit("📋 List Batch Jobs", style="bold magenta"))
    console.print()

    # Check for local job files (.job.json = pending/running, .done.json = completed)
    pending_files = glob.glob("results/*.job.json")
    done_files = glob.glob("results/*.job.done.json")
    all_job_files = pending_files + done_files

    if all_job_files:
        console.print("[bold cyan]📁 로컬 작업 파일:[/bold cyan]")
        kst_tz = timezone(timedelta(hours=9))  # noqa: F841

        # Sort by modification time (newest first)
        all_job_files_sorted = sorted(all_job_files, key=lambda x: Path(x).stat().st_mtime, reverse=True)[:10]

        for job_file in all_job_files_sorted:
            is_done = job_file.endswith(".done.json")
            try:
                with open(job_file, encoding="utf-8") as f:
                    job_data = json.load(f)
                job_name = job_data.get("job_name", "unknown")
                job_id = job_name.split("/")[-1] if "/" in job_name else job_name
                model = job_data.get("model", "N/A")
                num_requests = job_data.get("num_requests", 0)

                if is_done:
                    # Already completed - don't query API
                    console.print(f"  ✅ [DONE] {Path(job_file).name}")
                    console.print(f"     모델: {model} | 요청: {num_requests}개 | ID: {job_id}")
                else:
                    # Try to get status from API
                    try:
                        # Extract project and location from job name
                        # Format: projects/{project}/locations/{loc}/batchPredictionJobs/{id}
                        parts = job_name.split("/")
                        if len(parts) >= 4:
                            project_id = parts[1]
                            location = parts[3]  # Extract location from job name
                            client = genai.Client(vertexai=True, project=project_id, location=location)
                            job = client.batches.get(name=job_name)
                            state = str(job.state).replace("JobState.", "").replace("JOB_STATE_", "")

                            state_icons = {
                                "SUCCEEDED": "✅",
                                "FAILED": "❌",
                                "CANCELLED": "⚠️",
                                "RUNNING": "🔄",
                                "PENDING": "⏳",
                            }
                            icon = state_icons.get(state, "❓")
                            console.print(f"  {icon} [{state}] {Path(job_file).name}")
                            console.print(f"     모델: {model} | 요청: {num_requests}개 | ID: {job_id}")
                        else:
                            console.print(f"  ❓ {Path(job_file).name} (상태 확인 불가)")
                    except Exception as e:
                        console.print(f"  ❓ {Path(job_file).name} (API 조회 실패: {e})")
            except (OSError, json.JSONDecodeError):
                console.print(f"  ❓ {Path(job_file).name} (파일 읽기 실패)")

        console.print()

    # Interactive input for limit
    limit_str = Prompt.ask("[cyan]🔢 API에서 조회할 작업 수[/cyan]", default="10")
    try:
        limit = int(limit_str)
    except ValueError:
        limit = 10

    # Initialize client
    use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "FALSE").upper() == "TRUE"

    if use_vertex:
        project = os.getenv("GOOGLE_CLOUD_PROJECT")

        if not project:
            creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if creds_path and Path(creds_path).exists():
                try:
                    with open(creds_path, encoding="utf-8") as f:
                        creds_data = json.load(f)
                        project = creds_data.get("project_id")
                except (OSError, json.JSONDecodeError):
                    pass

        if not project:
            console.print("[red]❌ GOOGLE_CLOUD_PROJECT not set[/red]")
            raise typer.Exit(1)

        console.print(f"[cyan]Using Vertex AI (project: {project})[/cyan]")

        # Query both locations (global for gemini-3, us-central1 for older models)
        jobs: list = []
        for loc in ["global", "us-central1"]:
            try:
                client = genai.Client(vertexai=True, project=project, location=loc)
                loc_jobs = list(client.batches.list())
                jobs.extend(loc_jobs)
                console.print(f"[dim]  {loc}: {len(loc_jobs)}개 작업[/dim]")
            except Exception as e:
                console.print(f"[dim]  {loc}: 조회 실패 ({e})[/dim]")
    else:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            console.print("[red]❌ GOOGLE_API_KEY not set[/red]")
            raise typer.Exit(1)
        client = genai.Client(api_key=api_key)
        console.print("[cyan]Using Gemini Developer API[/cyan]")
        jobs = list(client.batches.list())

    console.print()

    try:
        # List batch jobs (already fetched above for Vertex AI)

        if not jobs:
            console.print("[yellow]No batch jobs found[/yellow]")
            return

        # Sort by create_time descending (most recent first)
        jobs_sorted = sorted(
            jobs,
            key=lambda j: j.create_time if hasattr(j, "create_time") and j.create_time else "",
            reverse=True,
        )[:limit]

        # Create jobs table
        from datetime import timedelta, timezone

        from rich.box import ROUNDED

        kst_tz = timezone(timedelta(hours=9))

        jobs_table = Table(
            title=f"📋 최근 배치 작업 ({len(jobs_sorted)}/{len(jobs)}개)",
            box=ROUNDED,
            border_style="cyan",
            show_lines=True,
        )
        jobs_table.add_column("#", style="dim", width=3, justify="right")
        jobs_table.add_column("상태", width=12, justify="center")
        jobs_table.add_column("모델", style="cyan", width=18)
        jobs_table.add_column("생성(KST)", width=12)
        jobs_table.add_column("Job ID", style="dim")

        state_display = {
            "SUCCEEDED": ("✅ 성공", "green"),
            "FAILED": ("❌ 실패", "red"),
            "CANCELLED": ("⚠️ 취소", "yellow"),
            "RUNNING": ("🔄 실행중", "cyan"),
            "PENDING": ("⏳ 대기중", "yellow"),
        }

        for idx, job in enumerate(jobs_sorted, 1):
            job_name = job.name if hasattr(job, "name") else "unknown"
            # Clean up state string (remove JobState. and JOB_STATE_ prefixes)
            state = str(job.state) if hasattr(job, "state") else "UNKNOWN"
            state = state.replace("JobState.", "").replace("JOB_STATE_", "")
            state_text, state_color = state_display.get(state, (state, "white"))
            model = str(job.model) if hasattr(job, "model") and job.model else "N/A"

            # Shorten model name
            if "/" in model:
                model = model.split("/")[-1]

            # Convert to KST
            created_kst = ""
            if hasattr(job, "create_time") and job.create_time:
                kst_time = job.create_time.astimezone(kst_tz)
                created_kst = kst_time.strftime("%m/%d %H:%M")

            # Extract job ID from full name (show full ID)
            job_id = job_name.split("/")[-1] if "/" in job_name else job_name

            jobs_table.add_row(
                str(idx),
                f"[{state_color}]{state_text}[/{state_color}]",
                model,
                created_kst,
                job_id,
            )

        console.print(jobs_table)
        console.print()

        # Show running jobs separately if any
        running_jobs = [
            j for j in jobs if str(j.state).replace("JobState.", "").replace("JOB_STATE_", "") in ("RUNNING", "PENDING")
        ]
        if running_jobs:
            console.print(f"[bold cyan]🔄 현재 실행 중인 작업: {len(running_jobs)}개[/bold cyan]")
            for rj in running_jobs:
                rj_state = str(rj.state).replace("JobState.", "").replace("JOB_STATE_", "")
                rj_id = rj.name.split("/")[-1] if "/" in rj.name else rj.name
                console.print(f"  • {rj_id} ({rj_state})")
            console.print()

        console.print("[dim]상세 정보: 메뉴에서 3번(status) 또는 4번(info) 선택[/dim]")

    except Exception as e:
        console.print(f"[red]❌ Error listing jobs: {e}[/red]")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1) from e


# =============================================================================
# Gemini Batch Main Menu Command
# =============================================================================


@app.command("gemini-batch")
def gemini_batch_menu() -> None:
    """Gemini Batch API (50% cost savings) - Interactive menu."""
    from rich.box import ROUNDED

    # Header
    header_content = "[bold white]🚀 Gemini Batch API[/bold white]\n[dim]50% 비용 절감 | 최대 24시간 소요[/dim]"
    console.print(Panel(header_content, style="magenta", box=ROUNDED))
    console.print()

    # Menu options with icons
    menu_table = Table(
        show_header=True,
        header_style="bold cyan",
        box=ROUNDED,
        border_style="cyan",
        padding=(0, 1),
    )
    menu_table.add_column("번호", justify="center", width=6)
    menu_table.add_column("기능", width=16)
    menu_table.add_column("설명", style="dim")

    menu_table.add_row("[bold green]1[/]", "🚀 infer", "이미지 배치 추론 시작")
    menu_table.add_row("[bold green]2[/]", "📥 resume", "배치 작업 결과 가져오기")
    menu_table.add_row("[bold green]3[/]", "📊 status", "작업 상태 확인")
    menu_table.add_row("[bold green]4[/]", "ℹ️  info", "상세 정보 조회")
    menu_table.add_row("[bold green]5[/]", "📋 list", "최근 작업 목록")
    menu_table.add_row("[bold green]6[/]", "🔧 parse", "결과 JSONL 파싱")
    menu_table.add_row("[bold red]0[/]", "🚪 exit", "종료")

    console.print(menu_table)
    console.print()

    # Get user choice
    choice = Prompt.ask(
        "[bold cyan]선택[/bold cyan]",
        choices=["0", "1", "2", "3", "4", "5", "6"],
        default="1",
    )

    console.print()

    # Execute selected option
    if choice == "0":
        console.print("[yellow]Exiting...[/yellow]")
        return
    elif choice == "1":
        _gemini_batch_infer()
    elif choice == "2":
        _gemini_batch_resume()
    elif choice == "3":
        _gemini_batch_status()
    elif choice == "4":
        _gemini_batch_info()
    elif choice == "5":
        _gemini_batch_list()
    elif choice == "6":
        _gemini_batch_parse()


def main() -> None:
    """Entry point for CLI."""
    # Show help if no arguments provided
    if len(sys.argv) == 1:
        sys.argv.append("--help")

    app()


if __name__ == "__main__":
    main()
