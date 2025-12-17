{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python3
    ruff
    pyright
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
