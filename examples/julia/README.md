# using Julia in AML

installation (the `apt` package is not available on the Ubuntu 22.04+ compute
images, so use the official juliaup installer, which is version-independent):
```
curl -fsSL https://install.julialang.org | sh -s -- --yes
source ~/.bashrc
julia -e "using Pkg; Pkg.add(\"IJulia\")"
```

You can now select Julia as the kernel in your notebook...
