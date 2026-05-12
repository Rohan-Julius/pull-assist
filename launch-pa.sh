#!/bin/bash
# pull-assist launcher — opens `pa` in a new VS Code integrated terminal
#
# Usage:
#   ./launch-pa.sh                    # just opens pa
#   ./launch-pa.sh review <URL>       # opens pa review in new terminal
#   ./launch-pa.sh status             # opens pa status in new terminal
#
# Install globally:
#   ln -s "$(pwd)/launch-pa.sh" /usr/local/bin/launch-pa

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PA_CMD="$SCRIPT_DIR/venv/bin/pa"

# Build the full pa command with arguments
if [ $# -eq 0 ]; then
    PA_FULL="$PA_CMD"
else
    PA_FULL="$PA_CMD $*"
fi

# Detect if we're inside VS Code
if [ -n "$TERM_PROGRAM" ] && [ "$TERM_PROGRAM" = "vscode" ]; then
    # Already in VS Code — use osascript to open a new integrated terminal
    # and run the command in it
    osascript -e '
    tell application "System Events"
        tell process "Code"
            -- Cmd+Shift+` opens a new terminal in VS Code
            keystroke "`" using {command down, shift down}
        end tell
    end tell'

    # Wait for terminal to open
    sleep 0.5

    # Type the command into the new terminal
    osascript -e "
    tell application \"System Events\"
        tell process \"Code\"
            keystroke \"clear; export PS1='pull-assist> '; $PA_FULL\"
            keystroke return
        end tell
    end tell"
else
    # Not in VS Code — try to open VS Code terminal, fall back to Terminal.app
    if command -v code &>/dev/null; then
        # Open VS Code with a new terminal running the command
        osascript -e '
        tell application "Visual Studio Code"
            activate
        end tell'
        sleep 0.3
        osascript -e '
        tell application "System Events"
            tell process "Code"
                keystroke "`" using {command down, shift down}
            end tell
        end tell'
        sleep 0.5
        osascript -e "
        tell application \"System Events\"
            tell process \"Code\"
                keystroke \"clear; export PS1='pull-assist> '; $PA_FULL\"
                keystroke return
            end tell
        end tell"
    else
        # Fallback: run in current terminal
        clear
        export PS1='pull-assist> '
        exec $PA_FULL
    fi
fi
