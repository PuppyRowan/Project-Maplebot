# Standard library imports
import asyncio
import io
import logging
import os
import random
import re
import sys
import time
import traceback
from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime, time as datetime_time, timedelta, timezone
from functools import wraps
from pathlib import Path
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    NoReturn,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

# Third-party imports
import aiohttp
import backoff
import discord
from aiohttp import web
from aiohttp.web import middleware
from aiohttp.client import ClientError  # Updated import
from discord import (
    FFmpegPCMAudio,
    HTTPException,
    Webhook,
    app_commands,
)
from discord.ext import commands, tasks
from discord.utils import get

# Local imports
from bot.config import (
    BOT_PREFIX,
    BARK_CHANCE,
    DEV_GUID,
    DEV_IDS,
    FEATURES_DISABLED,
    GAG_ACTIVE,
    PUPPY_TIME_CHANCE,
    TARGET_USER_ID,
    TOKEN,
    UWU_CHANCE,
    WEBHOOK_CACHE_TTL,
    WEBHOOK_NAME,
)
from bot.guild_settings import GuildSettingsManager
from bot.wordbanks import (
    BARK_VARIATIONS,
    BOT_SCOLDS,
    GAG_SOUNDS,
    MOOD_MESSAGES,
    MOODS,
    PUPPY_MESSAGES,
    SWEAR_REPLACEMENTS,
    SwearFilter,
    pronoun_replacer,
    MOOD_GAG_SOUNDS,
)
from bot.mood_analyzer import MoodAnalyzer
from bot.user_settings import UserSettingsManager

# Available moods from MOOD_MESSAGES 
MOODS = list(MOOD_MESSAGES.keys())

if not TOKEN:
    raise ValueError("No Discord token found in configuration!")

# Convert single ID to list if needed
TARGET_USER_IDS: List[int] = ([TARGET_USER_ID] if isinstance(TARGET_USER_ID, int) 
                             else list(TARGET_USER_ID))

if not TARGET_USER_IDS:
    raise ValueError("No target user IDs configured!")

DEV_MODE = False  # Default to dev mode on startup

async def clear_commands(bot: commands.Bot, guild_id: Optional[int] = None) -> None:
    """Clear all commands from global or guild scope"""
    try:
        if guild_id:
            # Clear guild commands
            guild = discord.Object(id=guild_id)
            bot.tree.clear_commands(guild=guild)
            await bot.tree.sync(guild=guild)
        else:
            # Clear global commands
            bot.tree.clear_commands(guild=None)  # Explicitly pass None for global clear
            await bot.tree.sync()
    except Exception as e:
        print(f"Error clearing commands: {e}")

@dataclass
class WebhookCacheEntry:
    """Webhook cache entry with TTL"""
    webhook: discord.Webhook
    expires: datetime

# Centralized logging configuration 
def setup_logging() -> Tuple[logging.Logger, Dict[str, logging.Logger]]:
    """Configure global logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('bot.log')
        ]
    )

    # Create the main bot logger
    logger = logging.getLogger('bot')
    logger.setLevel(logging.INFO)

    # Create sub-loggers
    loggers = {
        'bot.message': logging.getLogger('bot.message'),
        'bot.commands': logging.getLogger('bot.commands'),
        'bot.voice': logging.getLogger('bot.voice'),
        'bot.webhook': logging.getLogger('bot.webhook'),
        'bot.heartbeat': logging.getLogger('bot.heartbeat'),
        'bot.connection': logging.getLogger('bot.connection')
    }

    return logger, loggers

# Initialize logging
logger, loggers = setup_logging()

class HeartbeatManager:
    def __init__(self, url: str, timeout: int = 10):
        self.url = url
        self.timeout = timeout
        self.logger = loggers['bot.heartbeat']
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self) -> 'HeartbeatManager':
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self
        
    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        exc_tb: Optional[Any]
    ) -> None:
        if self.session:
            await self.session.close()
            
    async def send_heartbeat(self) -> bool:
        """Send heartbeat with proper error handling"""
        try:
            if not self.session:
                raise RuntimeError("Session not initialized")
                
            async with self.session.get(self.url) as response:
                if response.status == 200:
                    self.logger.debug("Heartbeat successful")
                    return True
                else:
                    self.logger.warning(f"Heartbeat failed with status {response.status}")
                    return False
                    
        except asyncio.TimeoutError:
            self.logger.error("Heartbeat timed out")
            return False
        except Exception as e:
            self.logger.error(f"Heartbeat error: {str(e)}")
            return False

class PuppySchedule:
    def __init__(self):
        self.schedule = {
            # Morning routine
            'breakfast': (datetime_time(6, 0), datetime_time(9, 0)),
            'morning_walk': (datetime_time(9, 0), datetime_time(11, 0)),
            # Midday
            'nap_time': (datetime_time(11, 0), datetime_time(14, 0)),
            'play_time': (datetime_time(14, 0), datetime_time(17, 0)),
            # Evening routine 
            'dinner': (datetime_time(17, 0), datetime_time(19, 0)),
            'evening_walk': (datetime_time(19, 0), datetime_time(21, 0)),
            'bedtime': (datetime_time(21, 0), datetime_time(23, 59)),
            'sleeping': (datetime_time(0, 0), datetime_time(6, 0))
        }

    def get_current_activity(self, current_time: Optional[datetime] = None) -> str:
        """Get current puppy activity with improved time handling"""
        if current_time is None:
            current_time = datetime.now(timezone.utc)
            
        current_time = current_time.time()
        current_minutes = current_time.hour * 60 + current_time.minute

        for activity, (start, end) in self.schedule.items():
            start_minutes = start.hour * 60 + start.minute
            end_minutes = end.hour * 60 + end.minute
            
            # Handle midnight crossing
            if end_minutes < start_minutes:  # Activity crosses midnight
                if current_minutes >= start_minutes or current_minutes <= end_minutes:
                    return activity
            else:  # Normal time range
                if start_minutes <= current_minutes <= end_minutes:
                    return activity

        return 'sleeping'  # Default activity

class CommandSyncManager:
    """Centralized command sync management"""
    def __init__(self, bot: commands.Bot, dev_mode: bool = False):
        self.bot = bot
        self.dev_mode = dev_mode
        self.logger = loggers['bot.commands']
        self._last_sync = None
        self._sync_lock = asyncio.Lock()

    async def _clear_commands(self, guild_id: Optional[int] = None) -> None:
        """Clear commands from specified scope"""
        try:
            if guild_id:
                guild = discord.Object(id=guild_id)
                self.bot.tree.clear_commands(guild=guild)
                await self.bot.tree.sync(guild=guild)
                self.logger.info(f"Cleared commands from guild {guild_id}")
            else:
                self.bot.tree.clear_commands(guild=None)
                await self.bot.tree.sync()
                self.logger.info("Cleared global commands")
        except Exception as e:
            self.logger.error(f"Error clearing commands: {e}")
            raise

    async def sync(self, mode: Optional[str] = None) -> bool:
        """
        Sync commands based on mode:
        - None: Use current dev_mode setting
        - 'dev': Sync to dev guild only
        - 'global': Sync globally
        - 'clear': Clear all commands
        """
        async with self._sync_lock:
            try:
                # Determine sync mode
                sync_mode = mode or ('dev' if self.dev_mode else 'global')
                
                if sync_mode == 'clear':
                    # Clear all commands
                    if DEV_GUID:
                        await self._clear_commands(DEV_GUID)
                    await self._clear_commands()
                    self.logger.info("Cleared all commands")
                    
                elif sync_mode == 'dev':
                    # Clear global first
                    await self._clear_commands()
                    
                    if DEV_GUID:
                        # Sync to dev guild
                        guild = discord.Object(id=DEV_GUID)
                        self.bot.tree.copy_global_to(guild=guild)
                        await self.bot.tree.sync(guild=guild)
                        self.logger.info(f"Synced commands to dev guild {DEV_GUID}")
                    else:
                        self.logger.warning("Dev mode but no DEV_GUID configured")
                        
                elif sync_mode == 'global':
                    # Clear dev guild first if exists
                    if DEV_GUID:
                        await self._clear_commands(DEV_GUID)
                        
                    # Sync globally
                    await self.bot.tree.sync()
                    self.logger.info("Synced commands globally")
                    
                else:
                    raise ValueError(f"Invalid sync mode: {sync_mode}")
                    
                self._last_sync = datetime.now()
                return True
                
            except Exception as e:
                error_msg = f"Command sync failed: {str(e)}"
                self.logger.error(error_msg)
                await notify_devs(self.bot, error_msg, f"Sync mode: {sync_mode}")
                return False

    @property
    def last_sync_time(self) -> Optional[datetime]:
        """Get last successful sync time"""
        return self._last_sync

class PuppyBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix=BOT_PREFIX, intents=intents)
        self.webhook_cache: Dict[int, WebhookCacheEntry] = {}
        self.guild_settings = GuildSettingsManager()
        self._cache_lock = asyncio.Lock()
        
        # Add status rotation
        self.status_rotation = None
        self.status_task = None
        self.schedule = PuppySchedule()
        
        # Add command sync manager
        self.sync_manager = CommandSyncManager(self, dev_mode=DEV_MODE)

        # Initialize web server
        self.web_server = WebServer()

        # Initialize status task attribute 
        self._status_task = None
        self._status_running = False

        self.mood_analyzer = MoodAnalyzer()
        self.user_settings = UserSettingsManager()

    async def sync_command_tree(self, guild_id: Optional[int] = None) -> None:
        """
        Sync command tree with proper error handling and cleanup
        """
        try:
            if guild_id:
                # Guild-specific sync
                guild = discord.Object(id=guild_id)
                # Clear existing guild commands
                self.tree.clear_commands(guild=guild)
                # Copy global commands to guild
                self.tree.copy_global_to(guild=guild)
                # Sync to guild
                await self.tree.sync(guild=guild)
                print(f"Commands synced to guild: {guild_id}")
            else:
                # Global sync
                # Clear all commands first
                self.tree.clear_commands()
                await self.tree.sync()
                print("Commands synced globally")
                
        except discord.HTTPException as e:
            print(f"Failed to sync commands: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error syncing commands: {e}")
            raise

    async def rotate_status(self) -> None:
        """Rotate through different status messages"""
        try:
            await self.wait_until_ready()
            self._status_running = True
            
            while not self.is_closed() and self._status_running:
                try:
                    # Generate fresh status messages
                    status_messages = self.generate_status_messages()
                    
                    # Rotate through each status
                    for message, activity_type in status_messages:
                        if not self._status_running:
                            break
                            
                        activity = discord.Activity(type=activity_type, name=message)
                        await self.change_presence(activity=activity)
                        await asyncio.sleep(300)  # 5 minute delay
                        
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in status rotation: {e}")
                    await asyncio.sleep(60)  # Wait before retry
                    
        finally:
            self._status_running = False

    async def start_status_rotation(self) -> None:
        """Start the status rotation task"""
        if self._status_task and not self._status_task.done():
            return
            
        self._status_task = self.loop.create_task(
            self.rotate_status(),
            name='status_rotation'
        )

    async def stop_status_rotation(self) -> None:
        """Stop the status rotation task"""
        self._status_running = False
        
        if self._status_task and not self._status_task.done():
            self._status_task.cancel()
            try:
                await self._status_task
            except asyncio.CancelledError:
                pass
            
        self._status_task = None

    async def setup_hook(self) -> None:
        """Initialize bot hooks and sync commands"""
        try:
            # Start web server
            await self.web_server.start()

            # Start status rotation with new method
            await self.start_status_rotation()
            
            # Start webhook cleanup task
            self.webhook_cleanup_task.start()
            
            # Perform initial command sync
            await self.sync_manager.sync()

        except Exception as e:
            logger.error(f"Error in setup: {e}")
            await self.cleanup()
            raise

    def generate_status_messages(self) -> List[Tuple[str, discord.ActivityType]]:
        """Generate list of status messages and their types"""
        messages = [
            (f"in {len(self.guilds)} servers! 🐾", discord.ActivityType.playing),
            (f"!puphelp for commands 📚", discord.ActivityType.listening),
            (f"feeling {self.guild_settings.get_settings(0).current_mood}! 🐕", discord.ActivityType.playing),
            ("your messages! 👀", discord.ActivityType.watching)
        ]
        return messages

    async def close(self) -> None:
        """Clean up tasks on bot shutdown"""
        try:
            # Stop status rotation first
            await self.stop_status_rotation()

            # Cleanup web server first
            if hasattr(self, 'web_server'):
                await self.web_server.cleanup()
            
            # Cancel existing tasks
            if hasattr(self, 'status_task') and self.status_task:
                self.status_task.cancel()
                
            if hasattr(self, 'webhook_cleanup_task') and self.webhook_cleanup_task.is_running():
                self.webhook_cleanup_task.cancel()
                
            # Call parent cleanup
            await super().close()
            
        except Exception as e:
            logger.error(f"Error during bot cleanup: {e}")
            raise
        
    async def send_as_puppy(
        self,
        channel: discord.TextChannel,
        content: str
    ) -> None:
        """Helper to send messages as the target user via webhook"""
        try:
            target_member = channel.guild.get_member(TARGET_USER_ID)
            if not target_member:
                await channel.send(content)
                return
                
            webhook = await self.get_webhook(channel)
            await webhook.send(
                content=content,
                username=target_member.display_name,
                avatar_url=target_member.display_avatar.url
            )
        except discord.HTTPException as e:
            print(f"Discord HTTP error in send_as_puppy: {e}")
            await channel.send(content)
        except Exception as e:
            print(f"Unexpected error in send_as_puppy: {e}")
            await channel.send(content)

    async def cleanup_webhook_cache(self) -> None:
        """Remove expired webhook cache entries"""
        now = datetime.now()
        async with self._cache_lock:
            expired = [k for k, v in self.webhook_cache.items() if v.expires <= now]
            for key in expired:
                del self.webhook_cache[key]

    async def get_webhook(
        self,
        channel: discord.TextChannel
    ) -> discord.Webhook:
        """Get or create webhook with caching and TTL"""
        now = datetime.now()
        
        # Check cache and validity
        if channel.id in self.webhook_cache:
            entry = self.webhook_cache[channel.id]
            if entry.expires > now:
                return entry.webhook
            
        try:
            # Get existing webhook or create new one
            webhooks = await channel.webhooks()
            webhook = discord.utils.get(webhooks, name=WEBHOOK_NAME)
            
            if webhook is None:
                webhook = await channel.create_webhook(name=WEBHOOK_NAME)
                
            # Update cache with new TTL
            self.webhook_cache[channel.id] = WebhookCacheEntry(
                webhook=webhook,
                expires=now + timedelta(seconds=WEBHOOK_CACHE_TTL)
            )
            return webhook
            
        except discord.HTTPException as e:
            print(f"Webhook error: {e}")
            raise

    def get_current_puppy_time(self) -> Optional[str]:
        """Get current activity using improved scheduler"""
        return self.schedule.get_current_activity()

    async def heartbeat_task(self) -> None:
        """Improved heartbeat task with backoff and jitter"""
        url = "https://mapletini.info/api/push/tTiQBg0SMf?status=up&msg=OK&ping="
        base_interval = 50
        max_interval = 300
        current_interval = base_interval

        async with HeartbeatManager(url) as hb:
            while not self.is_closed():
                try:
                    success = await hb.send_heartbeat()
                    
                    if success:
                        # Reset interval on success
                        current_interval = base_interval
                    else:
                        # Exponential backoff with max limit
                        current_interval = min(current_interval * 2, max_interval)

                    # Add jitter to prevent thundering herd
                    jitter = random.uniform(0.8, 1.2)
                    await asyncio.sleep(current_interval * jitter)
                    
                except Exception as e:
                    logger.error(f"Critical error in heartbeat task: {e}")
                    await asyncio.sleep(current_interval)

    @tasks.loop(minutes=10)  # Run every 10 minutes
    async def webhook_cleanup_task(self) -> None:
        """Periodic task to cleanup expired webhook cache entries"""
        try:
            logger.debug("Running webhook cache cleanup")
            await self.cleanup_webhook_cache()
            logger.debug(f"Webhook cache size after cleanup: {len(self.webhook_cache)}")
        except Exception as e:
            logger.error(f"Error in webhook cleanup task: {e}")

class WebServer:
    """Enhanced web server with security features and proper error handling"""
    def __init__(self, host: str = '0.0.0.0', port: int = 8080):
        self.host = host
        self.port = port
        self.app = web.Application(middlewares=[
            self.validate_request,
            self.rate_limit,
            self.add_security_headers
        ])
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        self.logger = loggers['bot.connection']
        
        # Rate limiting
        self.rate_limits: Dict[str, list] = defaultdict(list)
        self.max_requests = 30  # Max requests per minute
        self.rate_window = 60  # Window in seconds
        
        # Blocked IPs
        self.blocked_ips: set = set()
        self.block_duration = 300  # 5 minutes
        
        self._setup_routes()
        
    @middleware
    async def validate_request(self, request: web.Request, handler: Callable) -> web.Response:
        """Validate incoming requests"""
        try:
            # Get client IP
            ip = request.remote
            
            # Check if IP is blocked
            if ip in self.blocked_ips:
                return web.Response(status=403, text="IP blocked due to suspicious activity")
            
            # Validate request method
            if request.method not in {'GET', 'POST', 'HEAD'}:
                self._maybe_block_ip(ip)
                return web.Response(status=405, text="Method not allowed")
                
            # Check for malicious patterns
            path = request.path
            if self._is_suspicious_request(path):
                self._maybe_block_ip(ip)
                return web.Response(status=400, text="Invalid request")
                
            # Process valid request
            return await handler(request)
            
        except Exception as e:
            self.logger.error(f"Error in request validation: {e}")
            return web.Response(status=400, text="Bad Request")
            
    @middleware
    async def rate_limit(self, request: web.Request, handler: Callable) -> web.Response:
        """Apply rate limiting"""
        now = time.time()
        ip = request.remote
        
        # Clean old requests
        self.rate_limits[ip] = [t for t in self.rate_limits[ip] if t > now - self.rate_window]
        
        # Check rate limit
        if len(self.rate_limits[ip]) >= self.max_requests:
            return web.Response(status=429, text="Too Many Requests")
            
        # Add request timestamp
        self.rate_limits[ip].append(now)
        
        return await handler(request)
        
    @middleware
    async def add_security_headers(self, request: web.Request, handler: Callable) -> web.Response:
        """Add security headers to response"""
        response = await handler(request)
        
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Content-Security-Policy'] = "default-src 'self'"
        
        return response
        
    def _is_suspicious_request(self, path: str) -> bool:
        """Check for suspicious request patterns"""
        suspicious_patterns = [
            r'\.php$',
            r'wp-',
            r'\.asp',
            r'\.cgi',
            r'/admin',
            r'/shell',
            r'/config',
            r'\.\.',  # Path traversal
            r'[^a-zA-Z0-9/._-]'  # Invalid characters
        ]
        
        return any(re.search(pattern, path, re.IGNORECASE) for pattern in suspicious_patterns)
        
    def _maybe_block_ip(self, ip: str) -> None:
        """Add IP to blocked list"""
        self.blocked_ips.add(ip)
        self.logger.warning(f"Blocked suspicious IP: {ip}")
        
    def _setup_routes(self) -> None:
        """Configure web server routes"""
        # Static files
        static_path = Path('web/static')
        if static_path.exists():
            self.app.router.add_static('/static/', path=static_path)
            
        # Add routes
        self.app.router.add_get('/', self.handle_index)
        self.app.router.add_get('/health', self.handle_health)
        
    async def handle_index(self, request: web.Request) -> web.Response:
        """Serve index page"""
        try:
            index_path = Path('web/index.html')
            if index_path.exists():
                return web.FileResponse(index_path)
            return web.Response(
                text="Welcome to PuppyBot!",
                content_type='text/html'
            )
        except Exception as e:
            self.logger.error(f"Error serving index: {e}")
            return web.Response(
                text="Error serving page",
                status=500
            )
            
    async def handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint"""
        return web.json_response({
            "status": "ok",
            "timestamp": datetime.now().isoformat()
        })

    async def start(self) -> None:
        """Start the web server"""
        try:
            self.logger.info(f"Starting web server on {self.host}:{self.port}")
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            
            self.site = web.TCPSite(
                self.runner,
                self.host,
                self.port,
                ssl_context=None  # Add SSL context if needed
            )
            await self.site.start()
            
            self.logger.info(f"Web server running at http://{self.host}:{self.port}")
            
        except Exception as e:
            self.logger.error(f"Failed to start web server: {e}")
            await self.cleanup()
            raise

    async def cleanup(self) -> None:
        """Cleanup web server resources"""
        try:
            if self.site:
                await self.site.stop()
                self.site = None
                
            if self.runner:
                await self.runner.cleanup()
                self.runner = None
                
            self.logger.info("Web server shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during web server cleanup: {e}")

bot = PuppyBot()

@bot.event
async def on_ready():
    """Bot ready event handler with command sync"""
    print(f'Bot is ready as {bot.user}')
    
    try:
        # Start the heartbeat task
        bot.loop.create_task(bot.heartbeat_task())
        
        # Sync commands based on dev mode
        if DEV_MODE and DEV_GUID:
            # Clear global commands first
            bot.tree.clear_commands(guild=None)
            await bot.tree.sync()
            
            # Sync to dev guild
            guild = discord.Object(id=DEV_GUID)
            bot.tree.copy_global_to(guild=guild)
            await bot.tree.sync(guild=guild)
            print(f"Commands synced to dev guild: {DEV_GUID}")
        else:
            # Production mode - sync globally
            await bot.tree.sync()
            print("Commands synced globally")
            
    except discord.HTTPException as e:
        print(f"Failed to sync commands: {e}")
        await notify_devs(bot, str(e), "Command sync failed in on_ready")
    except Exception as e:
        print(f"Unexpected error during setup: {e}")
        await notify_devs(bot, str(e), "Setup failed in on_ready")

# Add this decorator at the top of main.py
def not_target_user():
    """Decorator to only block the target user from using commands"""
    async def predicate(ctx):
        if ctx.author.id in TARGET_USER_IDS:
            await ctx.send("*sad puppy noises* You can't use this command!")
            return False
        return True
    return commands.check(predicate)

def check_target_user_role():
    async def predicate(ctx):
        target_member = ctx.guild.get_member(TARGET_USER_ID)
        if not target_member:
            await ctx.send("*confused whimper* Can't find the target user!")
            return False
        
        if ctx.author.id in TARGET_USER_IDS:
            await ctx.send("*sad puppy noises* You can't use this command!")
            return False
            
        # Check if command user has higher role than target
        if not ctx.author.top_role > target_member.top_role:
            await ctx.send("*defiant bark* You need a higher role to do that!")
            return False
            
        return True
    return commands.check(predicate)


def create_hybrid_command(name: str, description: str):
    """Helper decorator to create hybrid commands that work as both slash and prefix commands"""
    def decorator(func):
        # Only register as hybrid command once
        @bot.hybrid_command(name=name, description=description)
        @wraps(func)  # Preserve function metadata
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        # Return the wrapper directly
        return wrapper
    return decorator

# Then replace each @bot.command() with the hybrid versions:

@create_hybrid_command(name='set_bark', description='Set probability of random barks (0-1)')
@app_commands.describe(chance="Probability between 0 and 1 (example: 0.05 = 5%)")
@not_target_user()
async def set_bark_chance(ctx, chance: float) -> None:
    """Set the probability of random barks (0-1)"""
    try:
        chance = float(chance)
        if not 0 <= chance <= 1:
            await bot.send_as_puppy(ctx.channel, '*confused whimper* Please give me a number between 0 and 1!')
            return
            
        settings = bot.guild_settings.get_settings(ctx.guild.id)
        settings.update_setting('bark_chance', chance)
        bot.guild_settings.save_settings()
        await bot.send_as_puppy(ctx.channel, f'*happy tail wags* Bark chance set to {chance*100:.1f}%!')
    except ValueError:
        await bot.send_as_puppy(ctx.channel, '*tilts head* That doesn\'t look like a valid number...')

@create_hybrid_command(name='set_uwu', description='Set probability of UwU speech (0-1)') 
@app_commands.describe(chance="Probability between 0 and 1 (example: 0.1 = 10%)")
@not_target_user()
async def set_uwu_chance(ctx, chance: float) -> None:
    """Set the probability of UwU speech (0-1)"""
    try:
        chance = float(chance)
        if not 0 <= chance <= 1:
            await ctx.send('❌ Please provide a number between 0 and 1')
            return
            
        settings = bot.guild_settings.get_settings(ctx.guild.id)
        settings.update_setting('uwu_chance', chance)
        bot.guild_settings.save_settings()
        await ctx.send(f'✅ UwU chance set to {chance*100:.1f}%')
    except ValueError:
        await ctx.send('❌ Please provide a valid number between 0 and 1')

@create_hybrid_command(name='gag', description='Toggle muzzle/gag mode on/off')
async def toggle_gag(ctx) -> None:
    """Toggle the puppy's muzzle on/off"""
    try:
        settings = bot.guild_settings.get_settings(ctx.guild.id)
        new_state = not settings.gag_active
        settings.update_setting('gag_active', new_state)
        bot.guild_settings.save_settings()
        
        if new_state:
            response = "*muffled whimpers as the muzzle is secured*"
        else:
            response = "*happy panting now that the muzzle is removed*"
            
        await bot.send_as_puppy(ctx.channel, response)
        
    except Exception as e:
        print(f"Error in gag command: {e}")
        await ctx.send("*whimpers* Something went wrong!")

@create_hybrid_command(name='off', description='Toggle extra features on/off')
async def toggle_features(ctx) -> None:
    """Toggle extra features on/off"""
    settings = bot.guild_settings.get_settings(ctx.guild.id)
    new_state = not settings.features_disabled
    settings.update_setting('features_disabled', new_state)
    bot.guild_settings.save_settings()
    
    if new_state:
        await bot.send_as_puppy(ctx.channel, "*calms down and acts more reserved*")
    else:
        await bot.send_as_puppy(ctx.channel, "*perks up and gets excited again*")

@create_hybrid_command(name='puppytime', description='Check current puppy schedule activity')
async def check_puppy_time(ctx) -> None:
    """Check what the puppy should be doing now"""
    current_activity = bot.get_current_puppy_time()
    if current_activity:
        activity_name = current_activity.replace('_', ' ')
        response = random.choice(PUPPY_MESSAGES[current_activity])
        await ctx.send(f"🕒 It's {activity_name} time! {response}")
    else:
        await ctx.send("💤 *sleepy snores* Zzz...")

@create_hybrid_command(name='puphelp', description='Show comprehensive command list and guide')
async def show_help(ctx) -> None:
    """Show comprehensive help information about the bot"""
    
    help_embed = discord.Embed(
        title="🐕 Puppy Bot User Guide",
        description="All commands work with both / and ! prefix\nExample: `/command` or `!command`",
        color=discord.Color.blue()
    )

    # Puppy Management Commands section
    puppy_management = """
**Puppy Management** (Admin Only)
• `addpuppy <user>` - Add a user as a puppy
• `removepuppy <user>` - Remove a user's puppy status
• `listpuppies` - Show all puppies in server
• `setpuppy <user> <setting> <value>` - Change puppy settings
• `puppysettings <user>` - View puppy's settings
    """
    help_embed.add_field(name="🎯 Puppy Management", value=puppy_management, inline=False)

    # Settings Commands section
    settings_cmds = """
**Probability Settings**
• `set_bark <0-1>` - Set random bark chance
↳ Example: `!set_bark 0.05` = 5% chance

• `set_uwu <0-1>` - Set UwU speech chance
↳ Example: `!set_uwu 0.1` = 10% chance

• `mood [new_mood]` - Check/set puppy's mood
↳ Available moods: happy, playful, sleepy, hungry, excited
↳ Example: `!mood playful`
    """
    help_embed.add_field(name="⚙️ Settings Commands", value=settings_cmds, inline=False)

    # Toggle Commands section
    toggle_cmds = """
• `gag` - Toggle muzzle/gag mode on/off
↳ Controls whether messages are muffled

• `off` - Toggle extra features
↳ Disables barks/UwU but keeps pronouns

• `click` - Play training clicker sound 🔆
↳ Only works when target is in voice channel
    """
    help_embed.add_field(name="🔄 Toggle & Sound Commands", value=toggle_cmds, inline=False)

    # Analysis Commands section
    analysis_cmds = """
• `analyze <text>` - Analyze text mood/tone
↳ Shows detected mood and confidence

• `puppytime` - Check current schedule activity
↳ Shows what puppy should be doing now

• `status` - Show bot status and settings
↳ Displays current configuration

• `ping` - Check bot latency
↳ Shows connection status
    """
    help_embed.add_field(name="📊 Analysis Commands", value=analysis_cmds, inline=False)

    # Current Settings Section - only show if in a guild
    if ctx.guild:
        settings = ctx.bot.guild_settings.get_settings(ctx.guild.id)
        current_settings = f"""
• Bark Chance: {settings.bark_chance*100:.1f}%
• UwU Chance: {settings.uwu_chance*100:.1f}%
• Current Mood: {settings.current_mood}
• Features: {'Disabled' if settings.features_disabled else 'Enabled'}
• Muzzle: {'Active' if settings.gag_active else 'Inactive'}
        """
        help_embed.add_field(name="📈 Server Settings", value=current_settings, inline=False)

    # Add tips section with more relevant information
    tips = """
• Most commands require higher role than target user
• Puppy management requires Administrator permission
• Sound commands only work in voice channels
• Settings are per-server and per-puppy
• Use `/status` for detailed bot status
• Use `/puppysettings` to view individual puppy settings
    """
    help_embed.add_field(name="💡 Tips & Notes", value=tips, inline=False)

    # Add footer with additional info
    help_embed.set_footer(text="Bot Version 2.0 • Made with ❤️ • Use /commands for command list")

    await ctx.send(embed=help_embed)

# Keep the old commands alias for backward compatibility
@create_hybrid_command(name='commands', description='Show command list (alias for puphelp)')
async def show_commands(ctx) -> None:
    """Alias for puphelp command"""
    await ctx.invoke(bot.get_command('puphelp'))

@bot.event
async def on_guild_join(guild):
    """Handle bot joining new guild by initializing settings"""
    # Create default settings for the new guild
    settings = bot.guild_settings.get_settings(guild.id)
    # Settings are auto-saved by get_settings() if they don't exist
    print(f"Initialized settings for new guild: {guild.name} (ID: {guild.id})")

# Add these helper functions after the imports
def apply_pronoun_replacements(text: str) -> str:
    """Apply pronoun replacements to text while preserving case and word boundaries"""
    return pronoun_replacer.replace(text)

def add_bark() -> str:
    """Generate a random bark message"""
    return random.choice(BARK_VARIATIONS)

def transform_to_uwu(text: str) -> str: 
    """Transform text to intensive UwU speech"""
    uwu_replacements = {
        'r': 'w',
        'l': 'w',
        'R': 'W', 
        'L': 'W',
        'th': 'd',
        'Th': 'D',
        'TH': 'D',
        'n': 'ny',
        'N': 'Ny',
        'ove': 'uv',
        'OVE': 'UV',
        'Ove': 'Uv',
        'you': 'yuw',
        'YOU': 'YUW',
        'You': 'Yuw',
        'ing': 'in',
        'ING': 'IN',
        'Ing': 'In',
        'es ': 'ez ',
        'ES ': 'EZ ',
        'Es ': 'Ez ',
    }
    
    emoticons = [
        'uwu', 'owo', '>w<', ':3', '~', '^w^', 'owo~', 'uwu~', '>w<~',
        '(｡♡‿♡｡)', '(◕ᴗ◕✿)', '(♡ω♡)', '(づ｡◕‿‿◕｡)づ', '(ﾉ◕ヮ◕)ﾉ*:･ﾟ✧',
        '(｀μ´)', '(∩˃o˂∩)', '♪(๑ᴖ◡ᴖ๑)♪', '(◡ ω ◡)'
    ]
    
    action_expressions = [
        '*nuzzles*', '*paws at*', '*wags tail*', '*purrs*', '*wiggles*',
        '*snuggles up*', '*boops*', '*bounces excitedly*'
    ]
    
    # Apply replacements
    modified = text
    for old, new in uwu_replacements.items():
        modified = modified.replace(old, new)
    
    # Split into words
    words = modified.split()
    modified_words = []
    
    for word in words:
        # Random stutter (20% chance)
        if len(word) > 2 and random.random() < 0.2:
            word = f"{word[0]}-{word}"
            
        # Random word emphasis (15% chance)
        if random.random() < 0.15:
            word = ''.join(c + '-' for c in word).rstrip('-')
            
        # Random elongation (25% chance)
        if random.random() < 0.25 and any(c in 'aeiou' for c in word):
            vowel_idx = random.choice([i for i, c in enumerate(word) if c in 'aeiou'])
            word = word[:vowel_idx] + word[vowel_idx] * random.randint(2, 4) + word[vowel_idx + 1:]
        
        modified_words.append(word)
    
    modified = ' '.join(modified_words)
    
    # Add random emoticons (50% chance)
    if random.random() < 0.5:
        modified += f" {random.choice(emoticons)}"
        
    # Add random action (30% chance)
    if random.random() < 0.3:
        modified += f" {random.choice(action_expressions)}"
        
    return modified

async def forward_attachments(
    webhook: discord.Webhook, 
    message: discord.Message
) -> List[discord.File]:
    """Forward message attachments via webhook"""
    files = []
    for attachment in message.attachments:
        try:
            # Download and prepare attachment
            file_bytes = await attachment.read()
            file = discord.File(
                fp=io.BytesIO(file_bytes),
                filename=attachment.filename,
                description=attachment.description
            )
            files.append(file)
        except Exception as e:
            print(f"Error processing attachment {attachment.filename}: {e}")
    return files

async def notify_devs(
    bot: PuppyBot,
    error_msg: str,
    context: str = "No context provided"
) -> None:
    """Send error notifications to developers"""
    from bot.config import DEV_IDS
    
    error_embed = discord.Embed(
        title="🚨 Bot Error",
        description=f"```py\n{error_msg[:2000]}```",  # Truncate to Discord's limit
        color=discord.Color.red(),
        timestamp=datetime.now()
    )
    error_embed.add_field(name="Context", value=context[:1024], inline=False)
    
    for dev_id in DEV_IDS:
        try:
            dev_user = await bot.fetch_user(dev_id)
            await dev_user.send(embed=error_embed)
        except Exception as e:
            print(f"Failed to notify developer {dev_id}: {e}")

def log_errors(
    func: Callable[..., Coroutine[Any, Any, Any]]
) -> Callable[..., Coroutine[Any, Any, Any]]:
    """Decorator to log errors and notify developers"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            # Get full traceback
            error_traceback = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            
            # Get context from args (assuming first arg is self/ctx/interaction)
            context = f"Function: {func.__name__}"
            if args and hasattr(args[0], 'guild'):
                context += f"\nGuild: {args[0].guild}"
            if args and hasattr(args[0], 'channel'):
                context += f"\nChannel: {args[0].channel}"
            
            # Get bot instance
            bot_instance = None
            if args and isinstance(args[0], commands.Context):
                bot_instance = args[0].bot
            elif args and isinstance(args[0], discord.Interaction):
                bot_instance = args[0].client
            elif args and isinstance(args[0], PuppyBot):
                bot_instance = args[0]
                
            if bot_instance:
                await notify_devs(bot_instance, error_traceback, context)
            
            # Re-raise the exception
            raise
    return wrapper

# Add to bot/main.py after the other helper functions

def fix_grammar(text: str) -> str:
    """
    Fix grammar while preserving the intent and puppy-like speech patterns
    No auto-capitalization or punctuation
    """
    # Split into sentences
    sentences = re.split(r'([.!?]+\s*)', text)
    fixed = []
    
    for i in range(0, len(sentences), 2):
        sentence = sentences[i]
        if not sentence.strip():
            continue
            
        # Get existing punctuation if present
        punct = sentences[i + 1] if i + 1 < len(sentences) else ''
        
        # Split into words
        words = sentence.split()
        
        # Handle "it" statements (from pronoun replacement)
        if len(words) >= 2:
            if words[0].lower() == 'it':
                # Fix common verb agreements
                if words[1].lower() in ['am', 'are']:
                    words[1] = 'is'
                elif words[1].lower() in ['have']:
                    words[1] = 'has'
                    
        # Reassemble sentence with existing punctuation only
        fixed.append(' '.join(words) + punct)
        
    return ''.join(fixed).strip()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('bot.message_handler')

# Import and initialize SwearFilter with default replacements
from bot.wordbanks import SWEAR_REPLACEMENTS
swear_filter = SwearFilter(SWEAR_REPLACEMENTS)

@bot.event
@log_errors 
async def on_message(message):
    message_logger = loggers['bot.message']
    message_logger.debug(f"Message received from {message.author.id}: {message.content}")
    
    # Don't respond to bot messages
    if (message.author.bot):
        return

    # Check if message is from our webhook
    if message.webhook_id:
        return
        
    # Process commands
    await bot.process_commands(message)

    # Check if message author is a puppy
    user_settings = bot.user_settings.get_settings(message.author.id)
    if not user_settings:
        await bot.process_commands(message)
        return

    # Get guild settings only for features_disabled check
    guild_settings = bot.guild_settings.get_settings(message.guild.id)
    
    # Analyze mood from message
    if len(message.content) > 3:
        new_mood, confidence = bot.mood_analyzer.analyze_mood(message.content)
        
        # Update user's mood if confidence is high enough
        if confidence > 0.3:
            user_settings.current_mood = new_mood
            bot.user_settings.save_settings()
            message_logger.debug(f"Updated mood to {new_mood} (confidence: {confidence:.2f})")

    try:
        files = await forward_attachments(message.channel, message)
        await message.delete()
    except Exception as e:
        logger.error(f"Error handling message: {e}")
        return

    # Process message content using user settings instead of guild settings
    lines = message.content.split('\n')
    modified_lines = []

    for line in lines:
        if not line.strip():
            modified_lines.append('')
            continue

        current_line = line
        filtered_line, was_filtered = swear_filter.filter_text(current_line)
        
        if was_filtered and filtered_line in BOT_SCOLDS:
            modified_lines.append(filtered_line)
            continue
        
        current_line = filtered_line

        # Use user's gag setting instead of guild's
        if not guild_settings.features_disabled and user_settings.gag_active:
            if user_settings.current_mood in MOOD_GAG_SOUNDS:
                current_line = random.choice(MOOD_GAG_SOUNDS[user_settings.current_mood])
            else:
                current_line = random.choice(MOOD_GAG_SOUNDS['neutral'])
            modified_lines.append(current_line)
            break
        else:
            words = []
            for word in current_line.split():
                if any(word.startswith(prefix) for prefix in ['<@', '<#', '<:', '<a:']):
                    words.append(word)
                else:
                    word = pronoun_replacer.replace(word)
                    
                    # Use user's UwU chance
                    if not guild_settings.features_disabled and random.random() < user_settings.uwu_chance:
                        word = transform_to_uwu(word)
                        
                    words.append(word)
            
            current_line = ' '.join(words)
            current_line = fix_grammar(current_line)
            modified_lines.append(current_line)

    modified_content = '\n'.join(modified_lines)

    # Apply user-specific features
    if not guild_settings.features_disabled and not user_settings.gag_active:
        prefix_lines = []
        
        # Use user's bark chance
        if random.random() < user_settings.bark_chance:
            bark = add_bark()
            prefix_lines.append(bark)
        
        # Use user's mood
        if user_settings.current_mood in MOOD_MESSAGES:
            mood_prefix = random.choice(MOOD_MESSAGES[user_settings.current_mood])
            prefix_lines.append(mood_prefix.rstrip())
        
        # Use user's puppy time chance
        current_activity = bot.get_current_puppy_time()
        if current_activity and random.random() < user_settings.puppy_time_chance:
            activity_message = random.choice(PUPPY_MESSAGES[current_activity])
            prefix_lines.append(activity_message)

        if prefix_lines:
            modified_content = '\n'.join(prefix_lines + ['']) + modified_content

    # Send modified message
    try:
        webhook = await bot.get_webhook(message.channel)
        await webhook.send(
            content=modified_content,
            username=message.author.display_name,
            avatar_url=message.author.display_avatar.url,
            files=files,
            allowed_mentions=discord.AllowedMentions(
                everyone=False,
                users=True,
                roles=True,
                replied_user=True
            )
        )
    except Exception as e:
        logger.error(f"Error sending webhook message: {e}")
        try:
            await message.channel.send(
                f"**{message.author.display_name}:** {modified_content}",
                files=files
            )
        except Exception as e2:
            logger.error(f"Error sending fallback message: {e2}")

# Add slash commands
@create_hybrid_command(name='bark', description='Make the puppy bark!')
async def bark(interaction: discord.Interaction) -> None:
    """Simple bark command"""
    await interaction.response.send_message(random.choice(BARK_VARIATIONS))

@create_hybrid_command(name='mood', description='Check or set puppy\'s current mood')
@app_commands.describe(new_mood="Optional: Set a new mood (happy/playful/sleepy/etc)")
async def mood(ctx, new_mood: Optional[str] = None) -> None:
    """Check or change the puppy's mood"""
    # Get guild ID safely from either Context or Interaction
    guild_id = ctx.guild.id if ctx.guild else None
    if not guild_id:
        await ctx.send("*whimpers* This command can only be used in a server!")
        return

    settings = bot.guild_settings.get_settings(guild_id)
    
    if new_mood:
        if new_mood.lower() not in MOODS:
            available_moods = ', '.join(MOODS)
            await ctx.send(f"*confused head tilt* Available moods are: {available_moods}")
            return
            
        settings.update_setting('current_mood', new_mood.lower())
        bot.guild_settings.save_settings()
        await ctx.send(f"*mood shifts* Now feeling {new_mood}!")
    else:
        current_mood = settings.current_mood
        await ctx.send(f"*tail wags* Currently feeling {current_mood}!")

@create_hybrid_command(name='status', description='Show current bot status and settings')  
async def status(interaction: discord.Interaction) -> None:
    """Show current bot status and settings"""
    settings = bot.guild_settings.get_settings(interaction.guild_id)
    
    embed = discord.Embed(
        title="🐕 Puppy Status",
        color=discord.Color.blue()
    )
    
    embed.add_field(
        name="Current Settings",
        value=f"""
        Mood: {settings.current_mood}
        Bark Chance: {settings.bark_chance*100:.1f}%
        UwU Chance: {settings.uwu_chance*100:.1f}%
        Features: {'Disabled' if settings.features_disabled else 'Enabled'}
        Muzzle: {'Active' if settings.gag_active else 'Inactive'}
        """,
        inline=False
    )
    
    current_activity = bot.get_current_puppy_time()
    embed.add_field(
        name="Schedule",
        value=f"Current Activity: {current_activity or 'None'}"
    )
    
    await interaction.response.send_message(embed=embed)

@create_hybrid_command(name='ping', description='Check bot latency and connection status') 
async def ping(ctx: commands.Context) -> None:
    """Check bot latency and response time"""
    # Create initial response embed
    embed = discord.Embed(
        title="🏓 Pong!",
        color=discord.Color.green()
    )

    # Get bot latency (heartbeat)
    bot_latency = round(bot.latency * 1000)  # Convert to ms
    
    # Add command processing time
    start_time = time.perf_counter()  # Now uses correct time.perf_counter
    message = await ctx.send(embed=embed)
    end_time = time.perf_counter()  # Now uses correct time.perf_counter
    api_latency = round((end_time - start_time) * 1000)  # Convert to ms
    
    # Update embed with latency information
    embed.add_field(
        name="Bot Latency",
        value=f"```{bot_latency}ms```",
        inline=True
    )
    embed.add_field(
        name="API Latency",
        value=f"```{api_latency}ms```",
        inline=True
    )
    
    # Add status indicator
    status = "🟢 Good" if bot_latency < 200 else "🟡 OK" if bot_latency < 400 else "🔴 High"
    embed.add_field(
        name="Status",
        value=status,
        inline=False
    )
    
    # Update the message with complete information
    await message.edit(embed=embed)

@create_hybrid_command(name='sync', description='Manage command sync (Dev Only)')
@commands.check(lambda ctx: ctx.author.id in DEV_IDS)
async def sync_commands(ctx: commands.Context, mode: str = "dev") -> None:
    """
    Sync commands based on mode:
    - dev: Sync to dev guild only
    - global: Sync globally
    - clear: Clear all commands
    """
    success = await ctx.bot.sync_manager.sync(mode)
    if success:
        await ctx.send(f"✅ Commands synced successfully ({mode} mode)")
    else:
        await ctx.send("❌ Command sync failed - check logs")

@create_hybrid_command(name='checkcommands', description='Check command registration (Dev Only)')
@commands.check(lambda ctx: ctx.author.id in DEV_IDS)
async def check_commands(ctx: commands.Context) -> None:
    """Debug command to check command registration"""
    # Get global commands
    global_commands = await ctx.bot.tree.fetch_commands()
    
    # Get guild commands if in a guild
    guild_commands = []
    if ctx.guild:
        guild_commands = await ctx.bot.tree.fetch_commands(guild=ctx.guild)
    
    embed = discord.Embed(title="Command Registration Status", color=discord.Color.blue())
    
    # Add global commands to embed
    global_cmd_list = "\n".join([f"• {cmd.name}" for cmd in global_commands]) or "No global commands"
    embed.add_field(name="Global Commands", value=global_cmd_list, inline=False)
    
    # Add guild commands to embed
    if guild_commands:
        guild_cmd_list = "\n".join([f"• {cmd.name}" for cmd in guild_commands]) or "No guild commands"
        embed.add_field(name=f"Guild Commands ({ctx.guild.name})", value=guild_cmd_list, inline=False)
    
    await ctx.send(embed=embed)

@create_hybrid_command(name='analyze', description='Analyze the mood of a message')
@app_commands.describe(text="Text to analyze")
async def analyze_mood(ctx, *, text: str) -> None:
    """Analyze the mood/tone of text"""
    mood, confidence = ctx.bot.mood_analyzer.analyze_mood(text)
    
    embed = discord.Embed(
        title="🎭 Mood Analysis",
        description=f"Text: {text}",
        color=discord.Color.blue()
    )
    
    embed.add_field(
        name="Detected Mood",
        value=f"{mood.title()} ({confidence*100:.1f}% confidence)",
        inline=False
    )
    
    await ctx.send(embed=embed)

class DiscordConnectionManager:
    def __init__(self, bot, max_retries: int = 5):
        self.bot = bot
        self.max_retries = max_retries
        self.logger = loggers['bot.connection']
        self.backoff_base = 1.5
        self.max_backoff = 60

    async def connect_with_backoff(self) -> None:
        """Connect with exponential backoff retry logic"""
        retries = 0
        last_error = None
        
        while retries < self.max_retries:
            try:
                # Attempt connection
                await self.bot.start()
                self.logger.info("Successfully connected to Discord")
                return
                
            except aiohttp.ClientConnectorDNSError as dns_err:
                last_error = dns_err
                self.logger.warning(f"DNS resolution failed: {dns_err}")
                
            except (discord.ConnectionClosed, 
                    aiohttp.ClientConnectorError, 
                    aiohttp.ClientConnectionError) as conn_err:
                last_error = conn_err
                self.logger.warning(f"Connection error: {conn_err}")
                
            except Exception as e:
                last_error = e
                self.logger.error(f"Unexpected error during connection: {e}")
            
            # Calculate backoff time
            backoff = min(
                self.backoff_base ** retries,
                self.max_backoff
            )
            
            self.logger.info(f"Retrying connection in {backoff:.1f} seconds...")
            await asyncio.sleep(backoff)
            retries += 1
            
        # If we get here, all retries failed
        raise ConnectionError(
            f"Failed to connect after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )

    async def start_bot(self) -> None:
        """Start bot with connection management"""
        while True:
            try:
                await self.connect_with_backoff()
                break
            except ConnectionError as e:
                self.logger.critical(f"Connection failed: {e}")
                # Wait before attempting full restart
                await asyncio.sleep(60)
                continue
            except Exception as e:
                self.logger.critical(f"Fatal error: {e}")
                raise

# Update run_bot.py to use connection manager
if __name__ == "__main__":
    connection_manager = DiscordConnectionManager(bot)
    try:
        asyncio.run(connection_manager.start_bot())
    except KeyboardInterrupt:
        print("\nBot shutdown initiated by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

# Add these constants after other constants
SOUNDS_DIR = os.path.join(os.path.dirname(__file__), 'sounds')
CLICKER_SOUND = os.path.join(SOUNDS_DIR, 'clicker.mp3')

# Add this helper function with other helper functions
async def play_sound_for_target(bot: commands.Bot, guild_id: int) -> bool:
    """
    Play clicker sound for first available target user in a voice channel.
    Returns True if sound was played successfully.
    """
    voice_logger = loggers['bot.voice']
    
    guild = bot.get_guild(guild_id)
    if not guild:
        voice_logger.error("Could not find guild")
        return False

    # Try each target user until we find one in a voice channel
    for target_id in TARGET_USER_IDS:
        target_member = guild.get_member(target_id)
        if target_member and target_member.voice and target_member.voice.channel:
            voice_channel = target_member.voice.channel
            # Rest of function remains the same...
            break
    else:  # No targets found in voice
        voice_logger.info("No target users found in voice channels")
        return False

    try:
        # Get or create voice client
        voice_client = guild.voice_client
        if voice_client:
            if voice_client.channel != voice_channel:
                await voice_client.move_to(voice_channel)
        else:
            voice_client = await voice_channel.connect(timeout=20, self_deaf=True)

        # Check if we successfully connected
        if not voice_client or not voice_client.is_connected():
            voice_logger.error("Failed to establish voice connection")
            return False

        # Set up audio source
        audio_source = discord.FFmpegPCMAudio(CLICKER_SOUND)

        def after_playing(error):
            if error:
                voice_logger.error(f"Error playing sound: {error}")
            # Use the bot's loop to schedule the cleanup
            bot.loop.create_task(cleanup_voice_client(voice_client))

        # Play the sound if not already playing
        if not voice_client.is_playing():
            voice_client.play(audio_source, after=after_playing)
            voice_logger.debug("Voice connection established")
            return True
        else:
            voice_logger.info("Already playing audio")
            return False

    except Exception as e:
        voice_logger.error(f"Error in play_sound_for_target: {e}")
        if 'voice_client' in locals() and voice_client:
            await cleanup_voice_client(voice_client)
        return False

async def cleanup_voice_client(voice_client: discord.VoiceClient) -> None:
    """Helper to cleanup voice client after playing"""
    try:
        if voice_client and voice_client.is_connected():
            await voice_client.disconnect()
    except Exception as e:
        print(f"Error cleaning up voice client: {e}")

# Add this command with other commands
@create_hybrid_command(name='click', description='Play training clicker sound')
@not_target_user()
async def click(ctx: commands.Context) -> None:
    """Play a clicker sound in target user's voice channel"""
    # Defer the response since we might need time to connect
    await ctx.defer()

    # Check if sound file exists
    if not os.path.exists(CLICKER_SOUND):
        await ctx.send("*sad whimper* Can't find my clicker...")
        return

    # Try to play sound
    success = await play_sound_for_target(ctx.bot, ctx.guild.id)
    
    if success:
        await ctx.send("*excitedly clicks the training clicker!* 🔆")
    else:
        # Check specific conditions for more informative error messages
        guild = ctx.bot.get_guild(ctx.guild.id)
        target_member = guild.get_member(TARGET_USER_ID[0])
        
        if not target_member:
            await ctx.send("*confused whimper* Can't find the puppy...")
        elif not target_member.voice:
            await ctx.send("*confused head tilt* The puppy needs to be in a voice channel first!")
        else:
            await ctx.send("*sad whine* Something went wrong trying to use the clicker...")

@create_hybrid_command(name='addpuppy', description='Add a user as a puppy')
@commands.has_permissions(administrator=True)
async def add_puppy(ctx, user: discord.Member):
    """Add a user as a puppy"""
    # Check if user is already a puppy
    if ctx.bot.user_settings.get_settings(user.id):
        await ctx.send(f"*tilts head* {user.display_name} is already a puppy!")
        return

    # Add new puppy
    ctx.bot.user_settings.add_puppy(user.id)
    await ctx.send(f"*happy tail wags* {user.display_name} is now a puppy! 🐕")

@create_hybrid_command(name='removepuppy', description='Remove a user from being a puppy')
@commands.has_permissions(administrator=True)
async def remove_puppy(ctx, user: discord.Member):
    """Remove a user from being a puppy"""
    if ctx.bot.user_settings.remove_puppy(user.id):
        await ctx.send(f"*sad whimpers* {user.display_name} is no longer a puppy...")
    else:
        await ctx.send(f"*confused head tilt* {user.display_name} wasn't a puppy to begin with!")

@create_hybrid_command(name='listpuppies', description='List all puppies in the server')
async def list_puppies(ctx):
    """List all puppies and their moods"""
    puppies = []
    for user_id, settings in ctx.bot.user_settings.settings.items():
        member = ctx.guild.get_member(int(user_id))
        if member:
            puppies.append(f"🐕 {member.display_name} - {settings.current_mood}")

    if puppies:
        embed = discord.Embed(
            title="🐕 Puppy List",
            description="\n".join(puppies),
            color=discord.Color.blue()
        )
        await ctx.send(embed=embed)
    else:
        await ctx.send("*sad whine* No puppies found!")

@create_hybrid_command(name='setpuppy', description='Change settings for a puppy')
@commands.has_permissions(administrator=True)
async def set_puppy_setting(ctx, user: discord.Member, setting: str, value: str):
    """Change settings for a specific puppy"""
    user_settings = ctx.bot.user_settings.get_settings(user.id)
    if not user_settings:
        await ctx.send(f"*confused whine* {user.display_name} isn't a puppy!")
        return

    try:
        if setting in ['bark_chance', 'uwu_chance', 'puppy_time_chance']:
            value = float(value)
            if not 0 <= value <= 1:
                await ctx.send("*tilts head* Chance must be between 0 and 1!")
                return
        elif setting in ['gag_active', 'features_disabled']:
            value = value.lower() == 'true'
        elif setting == 'current_mood':
            if value.lower() not in MOODS:
                await ctx.send(f"*confused* Available moods are: {', '.join(MOODS)}")
                return
            value = value.lower()
        else:
            await ctx.send(f"*confused whimper* Unknown setting: {setting}")
            return

        setattr(user_settings, setting, value)
        ctx.bot.user_settings.save_settings()
        await ctx.send(f"*happy tail wags* Updated {user.display_name}'s {setting} to {value}!")

    except ValueError:
        await ctx.send("*tilts head* That doesn't look like a valid value...")

@create_hybrid_command(name='puppysettings', description='Show settings for a puppy')
async def show_puppy_settings(ctx, user: discord.Member):
    """Show current settings for a puppy"""
    user_settings = ctx.bot.user_settings.get_settings(user.id)
    if not user_settings:
        await ctx.send(f"*confused whine* {user.display_name} isn't a puppy!")
        return

    embed = discord.Embed(
        title=f"🐕 {user.display_name}'s Settings",
        color=discord.Color.blue()
    )

    settings_text = f"""
    Mood: {user_settings.current_mood}
    Bark Chance: {user_settings.bark_chance*100:.1f}%
    UwU Chance: {user_settings.uwu_chance*100:.1f}%
    Puppy Time Chance: {user_settings.puppy_time_chance*100:.1f}%
    Muzzled: {'Yes' if user_settings.gag_active else 'No'}
    Features: {'Disabled' if user_settings.features_disabled else 'Enabled'}
    """
    embed.description = settings_text

    await ctx.send(embed=embed)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        logger.info("Starting bot...")
        bot.run(TOKEN)
    except Exception as e:
        logger.critical(f"Failed to start bot: {e}")
        raise