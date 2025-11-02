{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = {
    nixpkgs,
    flake-utils,
    ...
  }:
  flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = nixpkgs.legacyPackages.${system};
      lib = pkgs.lib;
      pyPkgs = pkgs.python313Packages;
      pyproject = builtins.fromTOML (builtins.readFile ./pyproject.toml);
      project = pyproject.project;
      fixString = x: lib.strings.toLower (builtins.replaceStrings ["_"] ["-"] x);
      getPkgs = x: lib.attrsets.attrVals (builtins.map fixString x) pyPkgs;
      package = pyPkgs.buildPythonPackage {
        pname = project.name;
        version = project.version;
        format = "pyproject";
        src = ./.;
        build-system = getPkgs pyproject.build-system.requires;
        dependencies = getPkgs project.dependencies ++ [ pkgs.ffmpeg-full ];
      };
      editablePackage = pyPkgs.mkPythonEditablePackage {
        pname = project.name;
        inherit (project) version scripts;
        root = "$PWD/src";
        dependencies = getPkgs project.optional-dependencies.dev or [];
      };
    in
    {
      packages.default = pyPkgs.toPythonApplication package;
      devShells.default = pkgs.mkShell {
        inputsFrom = [
          package
        ];
        buildInputs = [
          editablePackage
          pyPkgs.build
        ];
        shellHook = ''
          build-cython() { python setup.py build_ext -j 4 --inplace; }
        '';
      };
    }
  );
}
