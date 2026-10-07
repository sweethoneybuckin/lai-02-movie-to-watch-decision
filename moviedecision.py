import os
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

import aiohttp
import discord
from discord.ext import commands
from discord import app_commands
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
OMDB_API_KEY = os.getenv("OMDB_API_KEY")

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("MovieDecisionBot")
# Redam warning AFC bawaan google_genai agar logs bersih
logging.getLogger("google_genai").setLevel(logging.WARNING)

# Initialize Gemini Client with timeout (12 seconds)
gemini_client = None
if GEMINI_API_KEY:
    gemini_client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=types.HttpOptions(timeout=12000)
    )
else:
    logger.warning("GEMINI_API_KEY belum diset di .env")

# TMDB Base URLs
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"


@dataclass
class RecommendationItem:
    title: str
    year: Optional[str] = None
    media_type: str = "movie"  # 'movie' or 'tv'
    reason: str = ""
    tmdb_id: Optional[int] = None
    poster_url: Optional[str] = None
    genres: List[str] = field(default_factory=list)
    director: str = "-"
    cast: List[str] = field(default_factory=list)
    synopsis: str = ""
    providers_id: List[str] = field(default_factory=list)
    tomatometer: Optional[str] = None
    audience_score: Optional[str] = None
    imdb_rating: Optional[str] = None
    tmdb_rating: Optional[str] = None
    tmdb_url: Optional[str] = None


@dataclass
class UserSession:
    user_id: int
    original_prompt: str
    media_type_preference: str = "any"  # 'movie', 'tv', 'any'
    requested_ott: List[str] = field(default_factory=list)
    watched_titles: List[str] = field(default_factory=list)
    recommended_history: List[str] = field(default_factory=list)
    last_recommendation: Optional[RecommendationItem] = None


# Cache session per user
user_sessions: Dict[int, UserSession] = {}


# ==========================================
# 1. API HELPER FUNCTIONS (TMDB & OMDB)
# ==========================================

async def fetch_tmdb_details(session: aiohttp.ClientSession, title: str, media_type: str = "movie", year: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Mencari film/tv show di TMDB dan mengambil detail, credits, serta watch providers region ID."""
    if not TMDB_API_KEY:
        return None

    # Step 1: Search
    endpoint = f"{TMDB_BASE_URL}/search/{media_type}"
    params = {
        "api_key": TMDB_API_KEY,
        "query": title,
        "language": "id-ID",
        "include_adult": "false"
    }
    if year:
        if media_type == "movie":
            params["year"] = str(year)
        else:
            params["first_air_date_year"] = str(year)

    try:
        async with session.get(endpoint, params=params) as resp:
            if resp.status != 200:
                # Fallback search tanpa tahun / bahasa en-US jika tidak ketemu
                params.pop("year", None)
                params.pop("first_air_date_year", None)
                params["language"] = "en-US"
                async with session.get(endpoint, params=params) as resp2:
                    if resp2.status != 200:
                        return None
                    data = await resp2.json()
            else:
                data = await resp.json()

        results = data.get("results", [])
        if not results:
            # Coba cari tipe sebaliknya jika tidak ditemukan (misal user kira movie tapi ternyata TV series)
            alt_type = "tv" if media_type == "movie" else "movie"
            endpoint_alt = f"{TMDB_BASE_URL}/search/{alt_type}"
            params_alt = {"api_key": TMDB_API_KEY, "query": title, "include_adult": "false"}
            async with session.get(endpoint_alt, params=params_alt) as resp_alt:
                if resp_alt.status == 200:
                    alt_data = await resp_alt.json()
                    if alt_data.get("results"):
                        results = alt_data["results"]
                        media_type = alt_type

        if not results:
            return None

        best_match = results[0]
        item_id = best_match["id"]

        # Step 2: Ambil Full Detail (credits, watch providers, external_ids)
        detail_endpoint = f"{TMDB_BASE_URL}/{media_type}/{item_id}"
        detail_params = {
            "api_key": TMDB_API_KEY,
            "append_to_response": "credits,watch/providers,external_ids",
            "language": "id-ID"
        }
        async with session.get(detail_endpoint, params=detail_params) as detail_resp:
            if detail_resp.status != 200:
                # Fallback to en-US
                detail_params["language"] = "en-US"
                async with session.get(detail_endpoint, params=detail_params) as detail_resp2:
                    if detail_resp2.status != 200:
                        return None
                    full_detail = await detail_resp2.json()
            else:
                full_detail = await detail_resp.json()

        full_detail["detected_media_type"] = media_type
        return full_detail
    except Exception as e:
        logger.error(f"Error calling TMDB API: {e}")
        return None


async def fetch_omdb_ratings(session: aiohttp.ClientSession, imdb_id: Optional[str] = None, title: Optional[str] = None, year: Optional[str] = None) -> Dict[str, Optional[str]]:
    """Mengambil skor Rotten Tomatoes (Tomatometer & Audience), IMDb dari OMDb API."""
    result = {
        "tomatometer": None,
        "audience_score": None,
        "imdb_rating": None
    }
    if not OMDB_API_KEY:
        return result

    params = {"apikey": OMDB_API_KEY}
    if imdb_id:
        params["i"] = imdb_id
    elif title:
        params["t"] = title
        if year:
            params["y"] = str(year)
    else:
        return result

    try:
        async with session.get("http://www.omdbapi.com/", params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data.get("Response") == "True":
                    # Parse IMDb
                    imdb_val = data.get("imdbRating")
                    if imdb_val and imdb_val != "N/A":
                        result["imdb_rating"] = f"{imdb_val}/10"

                    # Parse Rotten Tomatoes dari list Ratings
                    for rating in data.get("Ratings", []):
                        if rating.get("Source") == "Rotten Tomatoes":
                            result["tomatometer"] = rating.get("Value")

                    # OMDb kadang tidak menyertakan audience score secara eksplisit,
                    # tapi beberapa data menyertakan boxoffice/metascore.
    except Exception as e:
        logger.error(f"Error calling OMDb API: {e}")

    return result


async def enrich_recommendation(rec: RecommendationItem) -> RecommendationItem:
    """Mengisi data visual & metadata TMDB + OMDb ke objek rekomendasi."""
    timeout = aiohttp.ClientTimeout(total=8)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tmdb_data = await fetch_tmdb_details(
            session=session,
            title=rec.title,
            media_type=rec.media_type,
            year=rec.year
        )

        if tmdb_data:
            rec.tmdb_id = tmdb_data.get("id")
            rec.media_type = tmdb_data.get("detected_media_type", rec.media_type)

            # Poster
            poster_path = tmdb_data.get("poster_path")
            if poster_path:
                rec.poster_url = f"{TMDB_IMAGE_BASE}{poster_path}"

            # Title & Year
            official_title = tmdb_data.get("title") or tmdb_data.get("name") or rec.title
            rec.title = official_title
            release_date = tmdb_data.get("release_date") or tmdb_data.get("first_air_date") or ""
            if release_date and len(release_date) >= 4:
                rec.year = release_date[:4]

            # Genres
            rec.genres = [g["name"] for g in tmdb_data.get("genres", [])]

            # Synopsis
            synopsis = tmdb_data.get("overview")
            if synopsis:
                rec.synopsis = synopsis[:300] + "..." if len(synopsis) > 300 else synopsis

            # TMDB rating
            vote_avg = tmdb_data.get("vote_average")
            if vote_avg:
                rec.tmdb_rating = f"{vote_avg:.1f}/10"

            # TMDB URL
            rec.tmdb_url = f"https://www.themoviedb.org/{rec.media_type}/{rec.tmdb_id}"

            # Credits: Director & Cast
            credits = tmdb_data.get("credits", {})
            crew = credits.get("crew", [])
            directors = [c["name"] for c in crew if c.get("job") == "Director"]
            if not directors and tmdb_data.get("created_by"):
                directors = [c["name"] for c in tmdb_data.get("created_by", [])]
            if directors:
                rec.director = ", ".join(directors[:2])

            cast_list = credits.get("cast", [])
            rec.cast = [c["name"] for c in cast_list[:4]]

            # Watch Providers (Region ID: Indonesia)
            providers_data = tmdb_data.get("watch/providers", {}).get("results", {}).get("ID", {})
            flatrate = providers_data.get("flatrate", [])
            rec.providers_id = [p["provider_name"] for p in flatrate]

            # External IDs (IMDb ID)
            imdb_id = tmdb_data.get("external_ids", {}).get("imdb_id")
            omdb_info = await fetch_omdb_ratings(session, imdb_id=imdb_id, title=rec.title, year=rec.year)
            rec.tomatometer = omdb_info.get("tomatometer")
            rec.audience_score = omdb_info.get("audience_score")
            rec.imdb_rating = omdb_info.get("imdb_rating")
        else:
            # Fallback OMDb langsung jika TMDB tidak ketemu
            omdb_info = await fetch_omdb_ratings(session, title=rec.title, year=rec.year)
            rec.tomatometer = omdb_info.get("tomatometer")
            rec.audience_score = omdb_info.get("audience_score")
            rec.imdb_rating = omdb_info.get("imdb_rating")

    return rec


# ==========================================
# 2. LLM REASONING ENGINE (GEMINI)
# ==========================================

async def get_ai_recommendation(session: UserSession, new_instruction: Optional[str] = None) -> Optional[RecommendationItem]:
    """Menggunakan Gemini untuk memilih kandidat rekomendasi terbaik berdasarkan prompt dan riwayat user."""
    if not gemini_client:
        logger.error("Gemini client belum terinisialisasi.")
        return None

    # Prompt Engineering
    system_instruction = (
        "Kamu adalah Movie & TV Series Concierge AI profesional yang bertugas membantu penonton memilih tontonan terbaik.\n"
        "Tugasmu:\n"
        "1. Analisis preferensi pengguna dari prompt (genre, mood, asal negara, tahun/era, kriteria penghargaan seperti Oscar, dsb).\n"
        "2. Perhatikan apakah pengguna meminta 'film' (movie), 'series/serial tv' (tv), atau bebas keduanya.\n"
        "3. Perhatikan jika pengguna menyebut layanan streaming/OTT tertentu (misal Netflix, Prime Video, HBO Max/Go, Disney+ Hotstar).\n"
        "   - Prioritaskan film/series yang kamu ketahui tersedia di OTT tersebut khusus untuk region INDONESIA.\n"
        "   - Jika pengguna TIDAK menyebutkan OTT sama sekali, kamu bebas merekomendasikan tontonan terbaik apapun meski tidak ada di OTT streaming.\n"
        "4. JANGAN merekomendasikan judul yang ada di daftar 'EXCLUDED_TITLES' (karena user sudah pernah nonton atau sudah ditolak).\n"
        "5. Output WAJIB berupa 1 (satu) objek JSON tunggal (BUKAN array/list [...], melainkan {...}) tanpa markdown triple backticks. Schema JSON:\n"
        "{\n"
        '  "title": "Judul resmi film/series",\n'
        '  "year": "Tahun rilis misal 2021",\n'
        '  "media_type": "movie" ATAU "tv",\n'
        '  "reason": "Penjelasan singkat 1-2 kalimat mengapa ini sangat cocok dengan selera & permintaan user",\n'
        '  "requested_ott": ["Netflix", ...],\n'
        '  "estimated_tomatometer": "90%"\n'
        "}\n"
    )

    excluded = session.watched_titles + session.recommended_history
    user_content = f"PROMPT UTAMA USER: {session.original_prompt}\n"
    if new_instruction:
        user_content += f"INSTRUKSI TAMBAHAN/REAKSI USER: {new_instruction}\n"
    if excluded:
        user_content += f"EXCLUDED_TITLES (JANGAN PILIH INI LAGI): {', '.join(excluded)}\n"

    # Prioritaskan model user atau model flash yang aktif & cepat
    preferred_model = GEMINI_MODEL or "gemini-3.5-flash-lite"
    candidate_models = [
        preferred_model,
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash",
        "gemini-3.6-flash",
        "gemini-3.8-flash"
    ]
    models_to_try = []
    for m in candidate_models:
        if m and m not in models_to_try:
            models_to_try.append(m)

    for model_name in models_to_try:
        try:
            logger.info(f"Mengirim request rekomendasi ke Gemini model: {model_name}...")
            # Panggil Gemini secara asynchronous dengan timeout ketat (12 detik per request)
            response = await asyncio.wait_for(
                gemini_client.aio.models.generate_content(
                    model=model_name,
                    contents=user_content,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        temperature=0.7
                    )
                ),
                timeout=12.0
            )

            response_text = response.text.strip() if response.text else ""
            if not response_text:
                logger.warning(f"Response kosong dari model {model_name}")
                continue

            # Bersihkan jika ada code block markdown
            if response_text.startswith("```"):
                lines = response_text.splitlines()
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                response_text = "\n".join(lines).strip()

            data = json.loads(response_text)

            # Jika model mengembalikan array/list [ {...} ], ambil elemen pertama
            if isinstance(data, list):
                if data and isinstance(data[0], dict):
                    data = data[0]
                else:
                    raise ValueError(f"Format JSON list tidak sesuai: {data}")

            if not isinstance(data, dict):
                raise ValueError(f"Format JSON bukan object/dict: {type(data)}")

            rec = RecommendationItem(
                title=data.get("title", ""),
                year=str(data.get("year", "")) if data.get("year") else None,
                media_type=data.get("media_type", "movie").lower(),
                reason=data.get("reason", "")
            )

            # Fallback tomatometer dari estimasi LLM jika API nantinya kosong
            est_tomato = data.get("estimated_tomatometer")
            if est_tomato:
                rec.tomatometer = str(est_tomato)

            # Update metadata session
            session.recommended_history.append(rec.title)
            logger.info(f"Berhasil mendapat rekomendasi: '{rec.title}' ({model_name})")
            return rec

        except asyncio.TimeoutError:
            logger.warning(f"Timeout (12s) saat memanggil Gemini model '{model_name}', mencoba model fallback...")
        except Exception as e:
            logger.warning(f"Gagal memanggil Gemini model '{model_name}': {e}, mencoba model fallback...")

    logger.error("Semua kandidat model Gemini gagal merespons.")
    return None


# ==========================================
# 3. DISCORD EMBED & VIEW (INTERACTIVE BUTTONS)
# ==========================================

def create_recommendation_embed(rec: RecommendationItem, user_prompt: str) -> discord.Embed:
    """Membuat Discord Rich Embed yang estetik dan informatif."""
    media_icon = "🎬" if rec.media_type == "movie" else "📺"
    type_str = "Film" if rec.media_type == "movie" else "TV Series"
    title_display = f"{media_icon} {rec.title}"
    if rec.year:
        title_display += f" ({rec.year})"

    embed = discord.Embed(
        title=title_display,
        url=rec.tmdb_url,
        description=f"*{rec.reason}*\n\n{rec.synopsis}" if rec.synopsis else f"*{rec.reason}*",
        color=discord.Color.from_rgb(229, 9, 20)  # Red / Netflix-like accent
    )

    # Info Fields
    if rec.genres:
        embed.add_field(name="🎭 Genre", value=", ".join(rec.genres), inline=True)
    embed.add_field(name="🏷️ Tipe", value=type_str, inline=True)

    if rec.director and rec.director != "-":
        embed.add_field(name="🎥 Sutradara", value=rec.director, inline=True)

    if rec.cast:
        embed.add_field(name="👥 Pemeran", value=", ".join(rec.cast), inline=False)

    # Streaming Providers di Region Indonesia
    if rec.providers_id:
        ott_text = " • ".join([f"**{p}**" for p in rec.providers_id])
    else:
        ott_text = "*Tidak tersedia di OTT streaming langganan ID (Mungkin tersedia di bioskop/sewa fisik/VOD)*"
    embed.add_field(name="🇮🇩 Tersedia di (Region Indonesia)", value=ott_text, inline=False)

    # Ratings (Tomatometer, IMDb, TMDB)
    rating_parts = []
    if rec.tomatometer:
        rating_parts.append(f"🍅 **Tomatometer:** {rec.tomatometer}")
    if rec.audience_score:
        rating_parts.append(f"🍿 **Audience Score:** {rec.audience_score}")
    if rec.imdb_rating:
        rating_parts.append(f"⭐ **IMDb:** {rec.imdb_rating}")
    elif rec.tmdb_rating:
        rating_parts.append(f"🌟 **TMDB:** {rec.tmdb_rating}")

    if rating_parts:
        embed.add_field(name="📊 Skor & Rating", value="  |  ".join(rating_parts), inline=False)

    # Poster image
    if rec.poster_url:
        embed.set_image(url=rec.poster_url)

    embed.set_footer(text=f"Permintaan: \"{user_prompt}\" • Gunakan tombol di bawah jika ingin opsi lain")
    return embed


class RecommendationView(discord.ui.View):
    """View berisi tombol interaktif: Rekomendasi Lain & Udah Nonton."""

    def __init__(self, session: UserSession):
        super().__init__(timeout=600)  # Aktif 10 menit
        self.session = session

        # Tambahkan Link TMDB Button jika ada URL
        if session.last_recommendation and session.last_recommendation.tmdb_url:
            self.add_item(
                discord.ui.Button(
                    label="Detail di TMDB",
                    url=session.last_recommendation.tmdb_url,
                    style=discord.ButtonStyle.link,
                    emoji="🔗"
                )
            )

    @discord.ui.button(label="Rekomendasi Lain", style=discord.ButtonStyle.secondary, emoji="🔄")
    async def another_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer(thinking=True)
        try:
            rec = await get_ai_recommendation(self.session, new_instruction="User minta rekomendasi judul lain yang berbeda.")
            if not rec:
                await interaction.followup.send("Maaf, tidak dapat menemukan rekomendasi baru saat ini. Silakan coba sesaat lagi.", ephemeral=True)
                return

            rec = await enrich_recommendation(rec)
            self.session.last_recommendation = rec

            embed = create_recommendation_embed(rec, self.session.original_prompt)
            new_view = RecommendationView(self.session)
            await interaction.followup.send(embed=embed, view=new_view)
        except Exception as e:
            logger.error(f"Error pada another_button: {e}", exc_info=True)
            try:
                await interaction.followup.send(f"Terjadi kesalahan saat memproses: {e}", ephemeral=True)
            except Exception:
                pass

    @discord.ui.button(label="Udah Pernah Nonton", style=discord.ButtonStyle.primary, emoji="✅")
    async def watched_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer(thinking=True)
        try:
            if self.session.last_recommendation:
                self.session.watched_titles.append(self.session.last_recommendation.title)

            rec = await get_ai_recommendation(
                self.session,
                new_instruction=f"User sudah pernah menonton '{self.session.last_recommendation.title if self.session.last_recommendation else ''}'. Berikan tontonan lain yang belum pernah ditonton!"
            )
            if not rec:
                await interaction.followup.send("Maaf, tidak dapat menemukan rekomendasi pengganti saat ini. Silakan coba sesaat lagi.", ephemeral=True)
                return

            rec = await enrich_recommendation(rec)
            self.session.last_recommendation = rec

            embed = create_recommendation_embed(rec, self.session.original_prompt)
            new_view = RecommendationView(self.session)
            await interaction.followup.send(
                content="Oke, noted! Film sebelumnya dicatat sudah ditonton. Ini rekomendasi berikutnya untukmu:",
                embed=embed,
                view=new_view
            )
        except Exception as e:
            logger.error(f"Error pada watched_button: {e}", exc_info=True)
            try:
                await interaction.followup.send(f"Terjadi kesalahan saat memproses: {e}", ephemeral=True)
            except Exception:
                pass


# ==========================================
# 4. DISCORD BOT SETUP & COMMANDS
# ==========================================

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)


@bot.event
async def on_ready():
    logger.info(f"Bot berhasil login sebagai {bot.user} (ID: {bot.user.id})")
    try:
        synced = await bot.tree.sync()
        logger.info(f"Berhasil mensinkronisasi {len(synced)} slash commands.")
    except Exception as e:
        logger.error(f"Gagal sync slash commands: {e}")


@bot.tree.command(name="tonton", description="Dapatkan rekomendasi film atau TV series sesuai seleramu")
@app_commands.describe(permintaan="Contoh: saran film komedi indonesia yang ada di netflix jangan jadul")
async def slash_tonton(interaction: discord.Interaction, permintaan: str):
    await interaction.response.defer(thinking=True)
    try:
        # Inisialisasi atau perbarui session user
        user_id = interaction.user.id
        session = UserSession(user_id=user_id, original_prompt=permintaan)
        user_sessions[user_id] = session

        # 1. Panggil Gemini AI
        rec = await get_ai_recommendation(session)
        if not rec:
            await interaction.followup.send("Waduh, terjadi kendala saat memproses rekomendasi AI. Silakan coba sesaat lagi.", ephemeral=True)
            return

        # 2. Enrich dengan data TMDB & OMDb (Poster, OTT ID, Tomatometer)
        rec = await enrich_recommendation(rec)
        session.last_recommendation = rec

        # 3. Render Embed & View Buttons
        embed = create_recommendation_embed(rec, permintaan)
        view = RecommendationView(session)
        await interaction.followup.send(embed=embed, view=view)
    except Exception as e:
        logger.error(f"Error handling slash_tonton: {e}", exc_info=True)
        try:
            await interaction.followup.send(f"Terjadi kesalahan saat memproses permintaan: {e}", ephemeral=True)
        except Exception:
            pass


@bot.command(name="tonton", aliases=["watch", "film", "rekomendasi"])
async def prefix_tonton(ctx: commands.Context, *, permintaan: str = ""):
    if not permintaan:
        await ctx.reply("Silakan sertakan kriteria tontonan yang kamu cari! Contoh:\n`!tonton saran film komedi indonesia yang ada di netflix dan jangan jadul`")
        return

    async with ctx.typing():
        try:
            user_id = ctx.author.id
            session = UserSession(user_id=user_id, original_prompt=permintaan)
            user_sessions[user_id] = session

            rec = await get_ai_recommendation(session)
            if not rec:
                await ctx.reply("Waduh, terjadi kendala saat memproses rekomendasi AI. Cek konfigurasi API key.")
                return

            rec = await enrich_recommendation(rec)
            session.last_recommendation = rec

            embed = create_recommendation_embed(rec, permintaan)
            view = RecommendationView(session)
            await ctx.reply(embed=embed, view=view)
        except Exception as e:
            logger.error(f"Error handling prefix_tonton: {e}", exc_info=True)
            await ctx.reply(f"Terjadi kendala saat memproses rekomendasi: {e}")


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    # Handle balasan chat natural seperti "udah nonton" atau "yang lain dong" jika mereferensi bot
    user_id = message.author.id
    content_lower = message.content.lower().strip()

    is_reply_to_bot = (
        message.reference
        and message.reference.resolved
        and message.reference.resolved.author.id == bot.user.id
    )
    is_mentioned = bot.user in message.mentions

    if (is_reply_to_bot or is_mentioned) and user_id in user_sessions:
        session = user_sessions[user_id]
        if any(w in content_lower for w in ["udah nonton", "udah pernah", "pernah nonton", "sudah nonton"]):
            if session.last_recommendation:
                session.watched_titles.append(session.last_recommendation.title)
            async with message.channel.typing():
                rec = await get_ai_recommendation(
                    session,
                    new_instruction=f"User sudah nonton '{session.last_recommendation.title if session.last_recommendation else ''}'. Carikan rekomendasi tontonan baru!"
                )
                if rec:
                    rec = await enrich_recommendation(rec)
                    session.last_recommendation = rec
                    embed = create_recommendation_embed(rec, session.original_prompt)
                    view = RecommendationView(session)
                    await message.reply(content="Siap, dicatat! Ini rekomendasi alternatifnya:", embed=embed, view=view)
                    return
        elif any(w in content_lower for w in ["yang lain", "ganti", "kurang suka", "next", "lain dong"]):
            async with message.channel.typing():
                rec = await get_ai_recommendation(session, new_instruction=f"User minta rekomendasi lain: {message.content}")
                if rec:
                    rec = await enrich_recommendation(rec)
                    session.last_recommendation = rec
                    embed = create_recommendation_embed(rec, session.original_prompt)
                    view = RecommendationView(session)
                    await message.reply(embed=embed, view=view)
                    return

    await bot.process_commands(message)


def main():
    if not DISCORD_TOKEN:
        print("[ERROR] DISCORD_TOKEN belum diset di file .env!")
        print("Silakan lengkapi file .env terlebih dahulu.")
        return

    bot.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
