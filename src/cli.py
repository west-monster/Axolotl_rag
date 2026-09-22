import argparse
import os
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from src.rag_engine import RAGEngine

console = Console()

def cmd_ingest(engine: RAGEngine, path: str):
    p = Path(path)
    if not p.exists():
        console.print(f"[bold red]Path '{path}' does not exist.[/bold red]")
        return
        
    with console.status(f"[bold green]Ingesting documents from '{path}'...", spinner="dots"):
        if p.is_dir():
            n = engine.ingest_directory(str(p))
        else:
            n = engine.ingest_file(str(p))
            
    console.print(f"[bold blue]Success![/bold blue] Indexed {n} chunks in the vector database.")

def cmd_stats(engine: RAGEngine):
    count = engine.vector_store.count()
    console.print(f"Total chunks in ChromaDB: [bold cyan]{count}[/bold cyan]")

def cmd_clear(engine: RAGEngine):
    engine.vector_store.clear()
    console.print("[bold red]The vector database has been cleared.[/bold red]")

def show_sources(sources):
    if not sources:
        return
    table = Table(title="Consulted Sources", show_header=True, header_style="bold magenta")
    table.add_column("Source", style="dim")
    table.add_column("Excerpt", width=70)
    table.add_column("Distance", justify="right")
    
    for doc, score in sources:
        source_name = os.path.basename(doc.metadata.get("source", "unknown"))
        excerpt = doc.page_content[:80].replace('\n', ' ') + "..." if len(doc.page_content) > 80 else doc.page_content.replace('\n', ' ')
        distance = f"{score:.4f}" if score is not None else "N/A"
        table.add_row(source_name, excerpt, distance)
        
    console.print(table)

def chat_loop(engine: RAGEngine):
    console.print(Panel.fit("[bold green]Welcome to Axolotl RAG[/bold green]\n"
                            "Type your question or use the following commands:\n"
                            "  [cyan]/ingest <path>[/cyan] - Index a file or directory\n"
                            "  [cyan]/stats[/cyan] - View DB statistics\n"
                            "  [cyan]/clear[/cyan] - Clear the DB\n"
                            "  [cyan]/exit[/cyan] or [cyan]/quit[/cyan] - Exit", 
                            title="Axolotl CLI", border_style="green"))
    
    while True:
        try:
            query = Prompt.ask("\n[bold yellow]You[/bold yellow]")
            if not query.strip():
                continue
                
            if query.startswith("/"):
                parts = query.split(maxsplit=1)
                cmd = parts[0].lower()
                if cmd in ["/exit", "/quit"]:
                    break
                elif cmd == "/ingest":
                    if len(parts) < 2:
                        console.print("[red]Usage: /ingest <path>[/red]")
                    else:
                        cmd_ingest(engine, parts[1])
                elif cmd == "/stats":
                    cmd_stats(engine)
                elif cmd == "/clear":
                    cmd_clear(engine)
                else:
                    console.print(f"[red]Unknown command: {cmd}[/red]")
                continue
                
            # Process RAG query
            console.print("\n[bold cyan]Axolotl:[/bold cyan] ", end="")
            response_stream, sources = engine.query_stream(query)
            
            full_response = ""
            for chunk in response_stream:
                if chunk:
                    console.print(chunk, end="")
                    full_response += chunk
            console.print("\n")
            
            # Display sources table
            show_sources(sources)

        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Axolotl RAG CLI")
    parser.add_argument("--ingest", type=str, help="Ingest a directory or file before starting")
    args = parser.parse_args()
    
    try:
        engine = RAGEngine()
        if args.ingest:
            cmd_ingest(engine, args.ingest)
        chat_loop(engine)
    except Exception as e:
        console.print(f"[bold red]Fatal Error:[/bold red] {e}")

if __name__ == "__main__":
    main()
