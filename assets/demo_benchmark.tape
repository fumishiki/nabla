# nabla benchmark demo — nabla vs PyTorch
# Run: vhs assets/demo_benchmark.tape

Output assets/demo_benchmark.gif
Set Shell "bash"
Set FontSize 15
Set Width 800
Set Height 600
Set Theme "Catppuccin Mocha"
Set WindowBar Colorful
Set Margin 20
Set MarginFill "#1E1E2E"
Set BorderRadius 8
Set TypingSpeed 60ms

Type "# nabla vs PyTorch — MLP training benchmark (GH200)"
Enter
Sleep 500ms

Type "cd benchmarks && cargo run --release --bin profile_train_graph --features cuda"
Enter
Sleep 15s

Type "python3 bench_pytorch.py"
Enter
Sleep 20s

Sleep 3s
