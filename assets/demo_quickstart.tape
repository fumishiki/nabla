# nabla REPL-style demo
# Run: vhs assets/demo_quickstart.tape
# NOTE: cargo build --example 00_demo --features cpu first for instant playback

Output assets/demo_quickstart.gif
Set Shell "bash"
Set FontSize 18
Set Width 740
Set Height 780
Set Theme "Dracula"
Set TypingSpeed 40ms
Set Padding 24

Sleep 500ms

Type "cargo run --example 00_demo --features cpu 2>/dev/null"
Enter

Sleep 3s
Sleep 4s
