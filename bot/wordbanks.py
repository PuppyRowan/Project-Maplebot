import re
import random
from typing import Tuple, List, Set, Dict
from difflib import get_close_matches
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.metrics.distance import edit_distance
from nltk.util import ngrams
from nltk import pos_tag

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

# First define the SWEAR_REPLACEMENTS dictionary
SWEAR_REPLACEMENTS = {
    # Basic swears
    'fuck': ['heck', 'frick', 'fluff', 'feck'],
    'shit': ['shoot', 'shucks', 'sugar', 'puppy mess', 'accident', 'slip'],
    'damn': ['darn', 'dang', 'drat', 'oh no', 'bad puppy'],
    'ass': ['butt', 'behind', 'bottom', 'tail', 'wiggly bits'],
    'bitch': ['meanie', 'puppy', 'doggo', 'good girl', 'brat'],
    'bastard': ['meanie', 'jerk', 'doodoo', 'ruffian', 'misbehaving pup'],
    'crap': ['crud', 'rubbish', 'stuff', 'oopsie', 'mess'],
    'cock': ['rooster', 'chicken', 'alpha', 'big bone', 'treat dispenser'],
    'pussy': ['kitten', 'soft spot', 'belly rubs', 'needy'],
    'dick': ['jerk', 'bad dog', 'alpha’s toy', 'plaything'],
    'cum': ['cream', 'puppy milk', 'special treat', 'messy fun'],
    'horny': ['excited', 'needy', 'wagging too much', 'panting hard'],
    'suck': ['lick', 'nuzzle', 'slurp', 'lap up'],
    'lick': ['clean', 'taste', 'groom', 'slurp'],
    'moan': ['whimper', 'whine', 'howl', 'needy noise'],
    'slut': ['enthusiastic pup', 'eager puppy', 'good toy', 'brave doggo'],
    'whore': ['obedient pup', 'needy thing', 'loyal puppy', 'begging dog'],
    'fucking': ['fluffing', 'ruffing', 'breeding', 'playing hard'],
    'screwed': ['messed up', 'in trouble', 'stuck in a crate', 'pawed too hard'],
    'jerk off': ['paw off', 'rub belly', 'get too excited'],
    'orgasm': ['happy tail', 'big moment', 'special treat', 'pup’s reward'],
    'spank': ['pat', 'smack', 'discipline', 'alpha’s lesson'],
    'dom': ['alpha', 'trainer', 'leader', 'top dog'],
    'sub': ['pup', 'good pet', 'obedient thing', 'waggy one'],
    'choke': ['tight collar', 'held firm', 'snug leash', 'gripped tight'],
    'gag': ['muzzle', 'silencer', 'puppy pacifier', 'toy in mouth'],
    'cunt': ['kitten', 'damp cave', 'hole', 'breed hole', 'moist'],
    
    # Compound swears
    'motherfucker': ['fluff monster', 'big bad alpha', 'breeder supreme'],
    'shithead': ['muzzle-brain', 'dumb pup', 'naughty tail'],
    'asshole': ['tailhole', 'bad dog', 'wiggly menace'],
    'dumbfuck': ['silly pup', 'airheaded doggo', 'bouncy idiot'],
    'cockslut': ['bone lover', 'treat chaser', 'alpha’s prize'],
    'pissbaby': ['needy pup', 'whimpering brat', 'potty training failure'],
    'fucktoy': ['good pup', 'alpha’s plaything', 'obedient pet'],
    'bastard dog': ['misbehaving mutt', 'untrained pup', 'wild stray'],
    'fuck me' : ['breed me', 'ruin me', 'wreck me']
}

BOT_SCOLDS = [
    "https://tenor.com/view/expletives-cursing-censored-gif-5138232",
    "Error 403: Forbidden Language Detected",
    "*whimpers* Unit has detected prohibited vocabulary",
    "https://tenor.com/view/hendo-hendoart-gif-20366498",
    "System Alert: Message violates wholesome guidelines",
    "https://tenor.com/view/censor-tv-glitch-cut-gif-16590916",
    "Illegal Operation: Bad Word Detected. Process terminated.",
    "https://tenor.com/view/excuse-my-foul-language-lindsey-graham-saturday-night-live-sorry-for-my-words-i-apologize-for-saying-those-words-gif-20430613",
    "⚠️ WARNING: Unwholesome content detected"
]

class SwearFilter:
    def __init__(self, replacements: dict, confidence_threshold: float = 0.7):
        self.replacements = replacements
        self.lemmatizer = WordNetLemmatizer()
        self.confidence_threshold = confidence_threshold
        
        # Compile word boundary patterns for each swear word
        self.swear_patterns = {}
        for swear in replacements.keys():
            # Create pattern that handles word boundaries and common substitutions
            pattern = r'\b' + re.escape(swear).replace('a','[a@4]').replace('i','[i1]').replace('o','[o0]').replace('e','[e3]') + r'\b'
            self.swear_patterns[swear] = re.compile(pattern, re.IGNORECASE)

    def filter_text(self, text: str) -> Tuple[str, bool]:
        """Filter text with improved word boundary detection"""
        if not text:
            return "", False
            
        try:
            was_filtered = False
            filtered = text.lower()  # Convert to lowercase for matching
            
            # First check for direct matches with word boundaries
            for swear, pattern in self.swear_patterns.items():
                if pattern.search(filtered):
                    replacements = self.replacements[swear]
                    replacement = random.choice(replacements)
                    filtered = pattern.sub(replacement, filtered)
                    was_filtered = True
            
            # Check for compound swears
            for compound in [k for k in self.replacements.keys() if ' ' in k]:
                if compound in filtered:
                    replacement = random.choice(self.replacements[compound])
                    filtered = filtered.replace(compound, replacement)
                    was_filtered = True

            return filtered, was_filtered
            
        except Exception as e:
            print(f"Error in filter_text: {e}")
            return text, False

# Finally initialize the filter
swear_filter = SwearFilter(SWEAR_REPLACEMENTS)

def filter_swears(text: str) -> Tuple[str, bool]:
    """Updated filter_swears function using new SwearFilter class"""
    return swear_filter.filter_text(text)

PUPPY_MESSAGES = {
    'breakfast': [
        "*excited bouncing* Breakfast time! Breakfast time!",
        "*nose boops food bowl* Time for morning nums!",
        "*tail wagging frantically* Food? Food? Food?",
        "*happy dancing* Is it breakfast yet?",
        "*gentle pawing at bowl* Hungy...",
        "*sits perfectly* Time for morning munchies?",
        "*stares intently at food area* Breakfast?",
        "*wiggles whole body* Morning feast time!"
    ],
    'morning_walk': [
        "*brings leash* Outside time?",
        "*excited prancing* Morning adventure?",
        "*zooms to door* Walk walk walk!",
        "*tail wagging* Time to explore!",
        "*happy circles* Outside time? Please?",
        "*nose boops door* Morning walkies?",
        "*stretches and looks hopeful* Adventure time?",
        "*playful bouncing* Let's go explore!"
    ],
    'nap_time': [
        "*curls up in sun spot* Sleepy time...",
        "*stretches* Getting kinda sleepy...",
        "*yawns big* Time for snoozy...",
        "*finds comfy spot* Nap time...",
        "*circles three times* Perfect nap spot...",
        "*sleepy tail wags* Getting drowsy...",
        "*cuddles into blanket* Snooze time...",
        "*flops over* Naptime now..."
    ],
    'play_time': [
        "*brings favorite toy* Play? Play? Play?",
        "*playful bounce* Time for games!",
        "*excited zoomies* Let's play!",
        "*tosses toy in air* Play with me?",
        "*playbow* Ready to have fun!",
        "*zooms around with toy* Chase me!",
        "*happy spins* Time for fun!",
        "*nudges toy closer* Wanna play?"
    ],
    'dinner': [
        "*excited spinning* Dinner time! Dinner time!",
        "*happy dance* Evening nums!",
        "*patient sitting* Food time again?",
        "*polite paw raise* Dinner please?",
        "*wiggles excitedly* Evening feast?",
        "*nose boops empty bowl* Time to eat?",
        "*perfect sit* Ready for dinner!",
        "*tail wagging intensifies* Food time?"
    ],
    'evening_walk': [
        "*gentle pawing* Evening walkies?",
        "Time for last outside adventure!",
        "*brings leash* One more walk?",
        "*hopeful look* Last adventure?",
        "*nose boops door* Outside again?",
        "*excited prancing* Evening stroll?",
        "*tail wags* Time to explore?",
        "*patient waiting by door* Walk time?"
    ],
        'bedtime': [
            "*sleepy yawn* Getting late...",
            "*drowsy tail wag* Bedtime soon?",
            "*cuddles close* Sleepy puppy time...",
            "*finds comfy spot* Ready for sleep...",
            "*gentle nesting* Time for bed?",
            "*sleepy stretches* Bedtime cuddles?",
            "*soft whimper* Tucked in time?",
            "*drowsy face* Sleep time now?"
        ]
    }

# Use list comprehension with tuple unpacking
BARK_VARIATIONS = [
    f"*{adj} bark* {sound}!" 
    for adj, sound in [
        ("excited", "woof woof"),
        ("playful", "bork bork"),
        ("happy", "arf arf"),
        ("sleepy", "woof... *yawn*")
    ]
] + ["awoooooo~", "wan wan!", "ruff ruff!"]

# Replace dynamic generation with static definitions
MOOD_MESSAGES = {
    'ecstatic': [
        '*bounces uncontrollably with joy* ',
        '*zooms around in pure happiness* ',
        '*can barely contain the excitement* '
    ],
    'excited': [
        '*tail wags wildly* ',
        '*bounces joyfully* ',
        '*spins excitedly* '
    ],
    'happy': [
        '*tail wags happily* ',
        '*prances around* ',
        '*smiles brightly* '
    ],
    'playful': [
        '*prances playfully* ',
        '*paws at toys* ',
        '*bounces energetically* '
    ],
    'content': [
        '*relaxes peacefully* ',
        '*wags tail gently* ',
        '*sighs contentedly* '
    ],
    'sleepy': [
        '*yawns softly* ',
        '*stretches lazily* ',
        '*blinks slowly* '
    ],
    'bored': [
        '*paws at ground listlessly* ',
        '*sighs dramatically* ',
        '*looks around for something to do* '
    ],
    'hungry': [
        '*stares at food bowl* ',
        '*paws at empty dish* ',
        '*whines at treat jar* '
    ],
    'anxious': [
        '*paces nervously* ',
        '*whimpers softly* ',
        '*tail tucked slightly* '
    ],
    'sad': [
        '*droops ears* ',
        '*lets out soft whine* ',
        '*tail droops low* '
    ],
    'meh': [
        '*shrugs slightly* ',
        '*flicks ears noncommittally* ',
        '*mild tail swish* '
    ],
    'neutral': [
        '*blinks calmly* ',
        '*sits quietly* ',
        '*observes surroundings* '
    ]
}

GAG_SOUNDS = [
    # Basic muffled sounds
    "*mmmphhh mmph!*",
    "*mmmf... mrrrfff...*",
    "*nnnngh... mmph...*",
    "*mrrrrph! mmmf!*",
    "*hnnngh... mphh!*",
    "*hrrmmph... mmff!*",

    # Struggle descriptions
    "*paws frantically at muzzle* mmmphhh!",
    "*wiggles nose against gag* nnnfff...",
    "*attempts to bark through gag* wrrrff!",
    "*tries to whine past muzzle* mnnnn...",
    "*tugs at restraints with gagged grumbles* mmph... nngh...",
    "*muffled grunts as tail flicks impatiently* hrrmff... mmph!",

    # Emotional reactions
    "*frustrated muffled noises* mmrph!",
    "*sad puppy eyes* mmmf...",
    "*pouty gagged sounds* hmmmph!",
    "*indignant muffled barking* brrrmph!",
    "*soft gagged sigh* hmmmf...",
    "*defeated whimper through gag* mmmnnn...",
    "*annoyed snort through muzzle* hnnnghf!",

    # Playful responses
    "*playful muffled boops* mff mff!",
    "*gagged tail wagging* mmrph mrph!",
    "*muffled happy sounds* mmmmff~",
    "*bouncy gagged noises* mph mph mph!",
    "*tries to giggle through gag* hmmhf! hmmphh!",

    # Complex actions
    "*pawing at muzzle while making puppy eyes* mmmf?",
    "*muffled zoomies with gagged barks* mrrf mrrf!",
    "*attempts to speak through gag* mmmrph mmph mmf!",
    "*gagged howling attempts* mmmroooooo...",
    "*shakes head, muffled protest* hnnmph! nnnghf!",
    "*squirms and wriggles, gag stifling playful whines* hnnnph mff!",

    # Protest sounds
    "*muffled protest whines* nnnngh!",
    "*frustrated gagged grumbles* mrrrmmph...",
    "*indignant muffled huffs* hmph! hmph!",
    "*gagged sighing* mmmffff...",
    "*growls softly through gag* hrmmmph!",
    "*loud gagged barking attempt* WRMMMPH! WRRFF!",
    "*weak huff of defeat* hnnph...",
    "*whines desperately against the gag* mmmnnnnph!",
]

MOOD_GAG_SOUNDS = {
    'ecstatic': [
        "*excited muffled bouncing* mmph mmph mmph!",
        "*gagged tail wagging intensifies* mrrrff~!",
        "*can barely contain muffled excitement* mmmf! mmmf!",
        "*happy gagged zoomies* mrrrf! mrrf! mrrf!"
    ],
    'excited': [
        "*muffled happy bouncing* mmrph mmph!",
        "*enthusiastic gagged noises* mff mff mff!",
        "*energetic tail wagging with gagged whines* mrrrfff~",
        "*playful gagged zoomies* mmph! mmph!"
    ],
    'happy': [
        "*content muffled sounds* mmmmff~",
        "*gagged tail wagging* mmrph mrph!",
        "*happy muffled purrs* mmmrrr~",
        "*cheerful gagged wiggles* mff mff!"
    ],
    'playful': [
        "*playful muffled boops* mff mff!",
        "*gagged play bow* mrrrff!",
        "*bouncy gagged noises* mph mph mph!",
        "*tries to giggle through gag* hmmhf! hmmphh!"
    ],
    'content': [
        "*relaxed muffled sighs* mmmmf~",
        "*gentle gagged humming* hmmm mmff~",
        "*peaceful muffled sounds* mff~",
        "*soft contented gag noises* mmrrr~"
    ],
    'sleepy': [
        "*sleepy muffled yawns* mmmmfff~",
        "*drowsy gagged mumbles* mmnnn...",
        "*tired muffled stretching* mmrphh~",
        "*lazy gagged sounds* mmmf..."
    ],
    'bored': [
        "*bored muffled sighs* mmmfff...",
        "*listless gagged sounds* meh mmph...",
        "*unenthusiastic gag noises* mmrph...",
        "*disinterested muffled huffs* hmph..."
    ],
    'hungry': [
        "*hungry muffled whines* mmmmnnn!",
        "*gagged food begging* mmph? mmph?",
        "*stomach growls with muffled whimpers* mrrrff...",
        "*impatient gagged food noises* mmf! mmf!"
    ],
    'anxious': [
        "*nervous muffled pacing* mmph... mmph...",
        "*worried gagged whimpers* mmnnn...",
        "*anxious pawing at muzzle* mmmf?",
        "*distressed muffled whines* nnnngh..."
    ],
    'sad': [
        "*sad muffled whimpers* mmmnn...",
        "*dejected gagged sighs* mmmmfff...",
        "*sorrowful muffled sounds* mrrff...",
        "*droopy gagged ears* mmph..."
    ],
    'meh': [
        "*indifferent muffled sounds* mmph.",
        "*noncommittal gagged noises* mff.",
        "*mild gagged reaction* mm.",
        "*barely interested muffled response* hm."
    ],
    'neutral': [
        "*calm muffled breathing* mmf...",
        "*steady gagged sounds* mmph.",
        "*composed muffled noises* mff.",
        "*neutral gagged response* mm."
    ]
}

MOODS = [
    'ecstatic',
    'excited', 
    'happy',
    'playful',
    'content',
    'neutral',
    'sleepy',
    'bored',
    'meh',
    'hungry',
    'anxious',
    'sad'
]

class VerbConjugator:
    def __init__(self):
        # Base verb mappings 
        self.base_verb_mappings = {
            'am': 'is',
            'was': 'was',
            'have': 'has',
            'do': 'does',
            'go': 'goes',
            'try': 'tries',
            'watch': 'watches',
            'buzz': 'buzzes',
            'fix': 'fixes',
            # Add this to prevent double conjugation
            'is': 'is'  
        }
        
        # Add handling for special cases
        self.special_cases = {
            'it is': 'it is',  # Prevent "it ises"
            'it has': 'it has',
            'it will': 'it will',
            'it would': 'it would'
        }

    def conjugate_verb(self, verb: str, tense: str = 'present') -> str:
        """Conjugate verb based on tense and special cases"""
        verb = verb.lower()

        # Check base mappings first
        if verb in self.base_verb_mappings:
            return self.base_verb_mappings[verb]

        # Check auxiliary verbs
        if verb in self.auxiliary_verbs:
            return self.auxiliary_verbs[verb]

        # Handle present tense conjugation
        if tense == 'present':
            # Apply suffix rules
            for pattern, replacement in self.verb_patterns.items():
                if pattern.search(verb):
                    return pattern.sub(replacement, verb)

        # Return original if no rules match
        return verb

    def validate_verb_agreement(self, subject: str, verb: str) -> str:
        """Validate subject-verb agreement"""
        subject = subject.lower()
        verb = verb.lower()

        # Handle "it" subject specifically
        if subject == 'it':
            # Special handling for common verbs
            if verb in self.base_verb_mappings:
                return self.base_verb_mappings[verb]
            
            # Handle auxiliary verbs
            if verb in self.auxiliary_verbs:
                return self.auxiliary_verbs[verb]

            # For regular verbs, ensure third person singular
            return self.conjugate_verb(verb, 'present')

        return verb

    def replace_verb_phrase(self, match: re.Match) -> str:
        """Replace entire verb phrases while maintaining agreement"""
        subject = match.group(1)
        verb = match.group(2)

        # Get the correct verb form
        conjugated = self.validate_verb_agreement(subject, verb)

        return f"{subject} {conjugated}"

    def replace(self, text: str) -> str:
        """Replace verbs while maintaining subject-verb agreement"""
        # First check for special cases
        for case, replacement in self.special_cases.items():
            if case in text.lower():
                return text
                
        # Then handle regular verb phrases
        verb_phrase_pattern = re.compile(
            r'\b(it|its)\s+(\w+)\b', 
            re.IGNORECASE
        )
        
        return verb_phrase_pattern.sub(
            self.replace_verb_phrase, 
            text
        )

# Update the pronoun replacer to use the new matcher
class PronounReplacement:
    def __init__(self):
        self.conjugator = VerbConjugator()
        self.pronoun_patterns = {
            # Handle contractions first - exact matches only
            r'\bI\'m\b': "it is",
            r'\bI\'ve\b': "it has", 
            r'\bI\'ll\b': "it will",
            r'\bI\'d\b': "it would",
            # Then handle basic pronouns
            r'\bI\b': "it",
            r'\bme\b': "it",
            r'\bmy\b': "its",
            r'\bmine\b': "its",
            r'\bmyself\b': "itself"
        }
        
        # Compile patterns and sort by length (longest first)
        self.compiled_patterns = sorted([
            (re.compile(pattern, re.IGNORECASE), replacement)
            for pattern, replacement in self.pronoun_patterns.items()
        ], key=lambda x: len(x[1]), reverse=True)

    def replace(self, text: str) -> str:
        if not text:
            return text

        # Apply replacements in order (longest first)
        for pattern, replacement in self.compiled_patterns:
            text = pattern.sub(replacement, text)
            
        # Clean up any double spaces
        text = ' '.join(text.split())
        
        return text

# Initialize the enhanced pronoun replacer
pronoun_replacer = PronounReplacement()

def apply_pronoun_replacements(text: str) -> str:
    """Apply enhanced pronoun replacements with proper grammar"""
    return pronoun_replacer.replace(text)
