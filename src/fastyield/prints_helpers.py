# ============================================================
# Pretty terminal print helpers
# ============================================================

# Text styles
RESET     = "\033[0m"
BOLD      = "\033[1m"
DIM       = "\033[2m"
ITALIC    = "\033[3m"
UNDERLINE = "\033[4m"
BLINK     = "\033[5m"
REVERSE   = "\033[7m"
STRIKE    = "\033[9m"

# ANSI basic
BLACK   = "\033[30m"
RED     = "\033[31m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
CYAN    = "\033[36m"
WHITE   = "\033[37m"

# ANSI bright
BBLACK   = "\033[90m"
BRED     = "\033[91m"
BGREEN   = "\033[92m"
BYELLOW  = "\033[93m"
BBLUE    = "\033[94m"
BMAGENTA = "\033[95m"
BCYAN    = "\033[96m"
BWHITE   = "\033[97m"

# 256-color extras
# 256-color extras


# Gray
GRAY        = "\033[38;5;245m"
LIGHTGRAY   = "\033[38;5;250m"
# Green
LIME        = "\033[38;5;118m"
SPRINGGREEN = "\033[38;5;48m"
OLIVEGREEN  = "\033[38;5;106m"
# Yellow/Orange
LEMON       = "\033[38;5;190m"
YELLOWSOFT  = "\033[38;5;221m"
AMBER       = "\033[38;5;214m"
ORANGE      = "\033[38;5;208m"
GOLD        = "\033[38;5;220m"
# Red/Pink
PINK        = "\033[38;5;213m"
SALMON      = "\033[38;5;209m"
CORALRED    = "\033[38;5;203m"
CRIMSONRED  = "\033[38;5;197m"
# Blue
TEAL        = "\033[38;5;44m"
SKYBLUE     = "\033[38;5;117m"
DEEPSKYBLUE = "\033[38;5;39m"
AZUREBLUE   = "\033[38;5;81m"
ROYALBLUE   = "\033[38;5;69m"
# Violet
VIOLET      = "\033[38;5;141m"
LAVENDER    = "\033[38;5;183m"

# Background text colors
BG_BLACK   = "\033[40m"
BG_RED     = "\033[41m"
BG_GREEN   = "\033[42m"
BG_YELLOW  = "\033[43m"
BG_BLUE    = "\033[44m"
BG_MAGENTA = "\033[45m"
BG_CYAN    = "\033[46m"
BG_WHITE   = "\033[47m"

def style(txt, *codes):
    return "".join(codes) + str(txt) + RESET

def print_header(title, sub=False):
    if sub:
        color = SKYBLUE
    else:
        color = DEEPSKYBLUE
    width = len(title) + 2
    print(style("╔" + "═" * width + "╗", color))
    print(style(f"║ {title} ║",          color))
    print(style("╚" + "═" * width + "╝", color))            

def print_subheader(title):
    print(style(f"{title}", BOLD, UNDERLINE, LIGHTGRAY))

def print_warning(msg):
    print(style(f"⚠ {msg}", BOLD, ITALIC, CRIMSONRED))

def print_info(msg, sub=False):
    if sub:
        print(style(f"{msg}", ITALIC, WHITE))
    else:
        print(style(f"{msg}", ITALIC, GOLD))
        
def print_time(msg):
    print(style(f"{msg}", ITALIC, LAVENDER))
    
def print_metric(label,                 key,            value,               units,
                 label_color=LIGHTGRAY, key_color=LIME, value_color=BYELLOW, units_color=LIGHTGRAY,
                 label_width=37,        key_width=19,   value_width=7,       units_width=20):

    label_txt = f"{label}:"
    key_txt   = f"{key}"
    value_txt = f"{value}"
    units_txt = f"{units}"

    label_pad = " " * max(0, label_width - len(label_txt))
    key_pad   = " " * max(0, key_width   - len(key_txt))
    value_pad = " " * max(0, value_width - len(value_txt))
    units_pad = " " * max(0, units_width - len(units_txt))

    print(
        "  "
        + style(label_txt, label_color)
        + label_pad
        + style(key_txt + key_pad, key_color)
        + " "
        + style("=", label_color)
        + " "
        + style(value_txt + value_pad, value_color)
        + "  "
        + style(units_txt + units_pad, units_color)
    )

def sci(x, nd=2):
    return f"{x:.{nd}e}"



























# TO PRINT PALETTE

COLORS = {
    "Gray": {
        "GRAY": GRAY,
        "LIGHTGRAY": LIGHTGRAY,
    },
    "Green": {
        "LIME": LIME,
        "SPRINGGREEN": SPRINGGREEN,
        "OLIVEGREEN": OLIVEGREEN,
    },
    "Yellow / Orange": {
        "LEMON": LEMON,
        "YELLOWSOFT": YELLOWSOFT,
        "AMBER": AMBER,
        "ORANGE": ORANGE,
        "GOLD": GOLD,
    },
    "Red / Pink": {
        "PINK": PINK,
        "SALMON": SALMON,
        "CORALRED": CORALRED,
        "CRIMSONRED": CRIMSONRED,
    },
    "Blue": {
        "TEAL": TEAL,
        "SKYBLUE": SKYBLUE,
        "DEEPSKYBLUE": DEEPSKYBLUE,
        "AZUREBLUE": AZUREBLUE,
        "ROYALBLUE": ROYALBLUE,
    },
    "Violet": {
        "VIOLET": VIOLET,
        "LAVENDER": LAVENDER,
    },
}


def color_text(text, color, bold=False):
    return f"{BOLD if bold else ''}{color}{text}{RESET}"


def extract_256_index(ansi_code):
    # "\033[38;5;141m" -> 141
    try:
        return int(ansi_code.split(";")[2].replace("m", ""))
    except Exception:
        return "?"


def print_palette():
    name_width   = 14
    code_width   = 8
    sample_width = 24

    print("\n" + "=" * 78)
    print("ANSI 256-COLOR PALETTE PREVIEW")
    print("=" * 78)
    print(f"{'NAME':<{name_width}} {'CODE':<{code_width}} {'NORMAL':<{sample_width}} {'BOLD':<{sample_width}}")
    print("-" * 78)

    for group_name, group in COLORS.items():
        print(f"\n[{group_name}]")
        for name, ansi in group.items():
            code = extract_256_index(ansi)
            normal_sample = color_text("██ SAMPLE TEXT ██", ansi, bold=False)
            bold_sample   = color_text("██ SAMPLE TEXT ██", ansi, bold=True)

            print(
                f"{name:<{name_width}} "
                f"{str(code):<{code_width}} "
                f"{normal_sample:<{sample_width + 10}} "
                f"{bold_sample:<{sample_width + 10}}"
            )

    print("\n" + "=" * 78 + "\n")


