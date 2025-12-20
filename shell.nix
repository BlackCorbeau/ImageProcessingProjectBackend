{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python3
    ruff
    pyright

    # Системные библиотеки для OpenCV и NumPy
    stdenv.cc.cc.lib
    zlib
    libGL
    libglvnd

    python3Packages.numpy
    python3Packages.opencv4
    python3Packages.flask
    python3Packages.scikit-learn
    python3Packages.scikit-image
    python3Packages.joblib
  ];

  shellHook = ''
    if [ ! -d ".venv" ]; then
      python -m venv .venv
    fi

    source .venv/bin/activate

    if [ -f "requirements-dev.txt" ]; then
      pip install -r requirements-dev.txt
    fi

    if [ -f "dependencies.txt" ]; then
      pip install -r dependencies.txt
    fi
  '';
}
