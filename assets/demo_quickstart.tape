# nabla REPL-style demo
# Run: vhs assets/demo_quickstart.tape
# NOTE: cargo build --example 00_demo --features cpu first for instant playback

Output assets/demo_quickstart.gif
Set Shell "bash"
Set FontSize 16
Set Width 800
Set Height 800
Set Theme "Catppuccin Mocha"
Set WindowBar Colorful
Set Margin 20
Set MarginFill "#1E1E2E"
Set BorderRadius 8
Set TypingSpeed 80ms

Sleep 500ms

Type "cargo run --example 00_demo --features cpu 2>/dev/null"
Enter

Sleep 15s
