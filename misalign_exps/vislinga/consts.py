# Wikimedia credentials are read from the environment (see the repo-root .env.example), e.g.
#   set -a; source ../.env; set +a
import os

# Optional auth (for higher rate limits): a Wikimedia bot password, see Special:BotPasswords.
_user, _bot_password = os.environ.get("WIKIMEDIA_USER"), os.environ.get("WIKIMEDIA_BOT_PASSWORD")
AUTH = (_user, _bot_password) if _user and _bot_password else None

# Descriptive User-Agent with contact details, as requested by Wikimedia API etiquette
USER_AGENT = (f"vislinga-dataset-tool/1.0 ({os.environ.get('WIKIMEDIA_CONTACT_EMAIL', 'your-email@example.com')})"
              " - fetching CC0/CC-BY images for research")

# Directory holding one sub-directory of images per concept (the VisLingA images)
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'images')
