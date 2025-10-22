"""Integrated CLI for car contamination classification."""

from pathlib import Path
from typing import Annotated, Optional

import typer

from src.api import VLMClient
from src.inference import run_batch_inference
from src.prompts import generate_prompt_template, save_transformed_guideline, transform_guideline_csv


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
    model: Annotated[str, typer.Option(help="Model name")] = "qwen3-vl-4b-instruct",
    max_tokens: Annotated[int, typer.Option(help="Maximum tokens to generate")] = 1000,
    temperature: Annotated[float, typer.Option(help="Sampling temperature")] = 0.0,
    output: Annotated[Optional[Path], typer.Option(help="Output file path (optional)")] = None,
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
    images_dir: Annotated[Path, typer.Argument(help="Directory containing images")],
    prompt: Annotated[Path, typer.Argument(help="Path to prompt file")],
    output: Annotated[Path, typer.Argument(help="Output CSV file path")],
    api_url: Annotated[
        str,
        typer.Option(help="VLM API endpoint URL"),
    ] = "http://vllm.mlops.socarcorp.co.kr/v1/chat/completions",
    model: Annotated[str, typer.Option(help="Model name")] = "qwen3-vl-4b-instruct",
    max_tokens: Annotated[int, typer.Option(help="Maximum tokens to generate")] = 1000,
    temperature: Annotated[float, typer.Option(help="Sampling temperature")] = 0.0,
    limit: Annotated[Optional[int], typer.Option(help="Maximum number of images to process (default: all)")] = None,
):
    """Run batch inference on multiple images."""
    typer.echo("=" * 60)
    typer.echo("Batch Inference")
    typer.echo("=" * 60)

    # Validate paths
    if not images_dir.exists():
        typer.echo(f"Error: Images directory not found: {images_dir}", err=True)
        raise typer.Exit(1)

    if not prompt.exists():
        typer.echo(f"Error: Prompt file not found: {prompt}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Images directory: {images_dir}")
    typer.echo(f"Prompt file: {prompt}")
    typer.echo(f"Output CSV: {output}")
    typer.echo(f"API URL: {api_url}")
    typer.echo(f"Model: {model}")
    typer.echo()

    try:
        summary = run_batch_inference(
            images_dir=images_dir,
            prompt_path=prompt,
            output_csv=output,
            api_url=api_url,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            limit=limit,
        )

        typer.echo("\n" + "=" * 60)
        typer.echo("Summary")
        typer.echo("=" * 60)
        typer.echo(f"Total images: {summary['total']}")
        typer.echo(f"Successful: {summary['successful']}")
        typer.echo(f"Failed: {summary['failed']}")
        typer.echo(f"Average latency: {summary['avg_latency']:.3f}s")
        typer.echo(f"Results saved to: {summary['output_path']}")

        typer.echo("\n" + "=" * 60)
        typer.echo("✓ Batch inference completed successfully!")
        typer.echo("=" * 60)

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        import traceback

        traceback.print_exc()
        raise typer.Exit(1) from e


@app.command("generate-prompt")
def generate_prompt(
    guideline: Annotated[Path, typer.Argument(help="Path to guideline CSV file")],
    output: Annotated[Optional[Path], typer.Argument(help="Output prompt file path (optional)")] = None,
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
