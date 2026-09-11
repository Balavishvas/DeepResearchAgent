from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import json

console = Console()

def format_message_content(message):
    """Convert message content to displayable string"""
    parts = []
    tool_calls_processed = False
    
    # Handle main content
    if isinstance(message.content, str):
        if message.content.strip():
            parts.append(message.content)
    elif isinstance(message.content, list):
        # Handle complex content like tool calls (Anthropic or Gemini format)
        for item in message.content:
            if isinstance(item, dict):
                if item.get('type') == 'text' and item.get('text'):
                    parts.append(item['text'])
                elif item.get('type') == 'tool_use':
                    parts.append(f"\n🔧 Tool Call: {item['name']}")
                    parts.append(f"   Args: {json.dumps(item.get('input', {}), indent=2)}")
                    parts.append(f"   ID: {item.get('id', 'N/A')}")
                    tool_calls_processed = True
            elif hasattr(item, 'text'):
                parts.append(item.text)
            else:
                parts.append(str(item))
    elif message.content:
        parts.append(str(message.content))
    
    # Handle tool calls attached to the message (OpenAI / LangChain format)
    if not tool_calls_processed and hasattr(message, 'tool_calls') and message.tool_calls:
        for tool_call in message.tool_calls:
            parts.append(f"\n🔧 Tool Call: {tool_call['name']}")
            parts.append(f"   Args: {json.dumps(tool_call.get('args', {}), indent=2)}")
            parts.append(f"   ID: {tool_call.get('id', 'N/A')}")
    
    return "\n".join(parts)


def format_messages(messages):
    """Format and display a list of messages with Rich formatting"""
    for m in messages:
        msg_type = m.__class__.__name__.replace('Message', '')
        content = format_message_content(m)

        if msg_type == 'Human':
            console.print(Panel(content, title="🧑 Human", border_style="blue"))
        elif msg_type in ('Ai', 'AI'):
            console.print(Panel(content, title="📝 AI", border_style="bright_black"))
        elif msg_type == 'Tool':
            console.print(Panel(content, title="🔧 Tool Output", border_style="yellow"))
        else:
            console.print(Panel(content, title=f"📝 {msg_type}", border_style="white"))


def format_message(messages):
    """Alias for format_messages for backward compatibility"""
    return format_messages(messages)


def show_prompt(prompt_text: str, title: str = "Prompt", border_style: str = "blue"):
    """
    Display a prompt with rich formatting and XML tag highlighting.
    
    Args:
        prompt_text: The prompt string to display
        title: Title for the panel (default: "Prompt")
        border_style: Border color style (default: "blue")
    """
    formatted_text = Text(prompt_text)
    formatted_text.highlight_regex(r'<[^>]+>', style="bold blue")
    formatted_text.highlight_regex(r'##[^#\n]+', style="bold magenta")
    formatted_text.highlight_regex(r'###[^#\n]+', style="bold cyan")

    console.print(Panel(
        formatted_text, 
        title=f"[bold green]{title}[/bold green]",
        border_style=border_style,
        padding=(1, 2)
    ))
