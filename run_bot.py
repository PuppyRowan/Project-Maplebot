import sys
import asyncio
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from bot.main import bot
from bot.config import TOKEN

async def main():
    try:
        await bot.start(TOKEN)
    except KeyboardInterrupt:
        await bot.close()
    finally:
        if not bot.is_closed():
            await bot.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot shutdown initiated by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)