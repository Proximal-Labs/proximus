;; Keep the language server and save hook local to this project.
((nil . ((eval . (when (and (derived-mode-p 'python-mode 'python-ts-mode)
                           (not (file-remote-p default-directory)))
                  (let ((root (locate-dominating-file default-directory
                                                     ".dir-locals.el")))
                    (load (expand-file-name "scripts/ruff-on-save.el" root)
                          nil t)
                    (proximus-enable-ruff-on-save root))))))
 (python-mode
  . ((eglot-server-programs
      . (((python-mode python-ts-mode)
          . ("uv" "run" "--locked" "pyright-langserver" "--stdio"))))))
 (python-ts-mode
  . ((eglot-server-programs
      . (((python-mode python-ts-mode)
          . ("uv" "run" "--locked" "pyright-langserver" "--stdio")))))))
