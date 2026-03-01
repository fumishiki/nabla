# nabla quick start demo — 5 lines of GPU math
# Run: vhs assets/demo_quickstart.tape

Output assets/demo_quickstart.gif
Set Shell "bash"
Set FontSize 16
Set Width 860
Set Height 500
Set Theme "Dracula"
Set TypingSpeed 30ms
Set Padding 20

Type "cat nabla-ml/examples/00_demo.rs"
Enter
Sleep 2s

Type "cargo run --example 00_demo --features cpu"
Enter
Sleep 8s

Sleep 2s
