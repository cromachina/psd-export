{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    poetry2nix.url = "github:nix-community/poetry2nix";
  };
  outputs = {
    self,
    nixpkgs,
    flake-utils,
    poetry2nix,
  }:
  flake-utils.lib.eachDefaultSystem (
    system:
    let
      pkgs = nixpkgs.legacyPackages.${system};
      inherit (poetry2nix.lib.mkPoetry2Nix { inherit pkgs; }) mkPoetryApplication mkPoetryEnv;
      config = {
        projectDir = self;
        preferWheels = true;
        nativeBuildInputs = with pkgs; [
          python312Packages.distutils
          python312Packages.cython
        ];
        configurePhase = ''
          runHook preConfigure
          C_INCLUDE_PATH=${pkgs.python312Packages.numpy.coreIncludeDir} cythonize -i '**/*.pyx'
          runHook postConfigure
        '';
      };
      pythonldlibpath = pkgs.lib.makeLibraryPath (with pkgs; [
        stdenv.cc.cc
        libGL
        glib
      ]);
    in
    {
      packages.default = mkPoetryApplication config;
      devShells.default = pkgs.mkShell {
        packages = with pkgs; [
          poetry
          (mkPoetryEnv { projectDir = self; preferWheels = true; })
        ] ++ config.nativeBuildInputs;
        shellHook = ''
          export LD_LIBRARY_PATH=${pythonldlibpath}
        '';
      };
    }
  );
}
