;;; ruff-on-save.el --- Local Ruff save hook -*- lexical-binding: t; -*-

(defvar-local proximus-ruff-root nil)

(defun proximus-ruff-before-save ()
  "Apply safe Ruff fixes and formatting without changing the file on disk."
  (when (and buffer-file-name (not (file-remote-p buffer-file-name)))
    (let ((source (current-buffer))
          (filename buffer-file-name)
          (default-directory proximus-ruff-root)
          (ruff (expand-file-name ".venv/bin/ruff" proximus-ruff-root))
          (errors (make-temp-file "proximus-ruff-")))
      (unwind-protect
          (with-temp-buffer
            (insert (with-current-buffer source
                      (save-restriction
                        (widen)
                        (buffer-substring-no-properties (point-min) (point-max)))))
            (dolist (args '(("check" "--fix-only" "--no-unsafe-fixes")
                            ("format")))
              (let ((status (apply #'call-process-region
                                   (point-min) (point-max) ruff t
                                   (list t errors) nil
                                   (append args (list "--stdin-filename" filename "-")))))
                (unless (eq status 0)
                  (error "Ruff failed: %s"
                         (with-temp-buffer
                           (insert-file-contents errors)
                           (buffer-string))))))
            (let ((formatted (current-buffer)))
              (with-current-buffer source
                (save-restriction
                  (widen)
                  (replace-buffer-contents formatted)))))
        (delete-file errors)))))

(defun proximus-enable-ruff-on-save (root)
  "Enable Ruff for this Python buffer in ROOT."
  (setq-local proximus-ruff-root root)
  (add-hook 'before-save-hook #'proximus-ruff-before-save nil t))

(provide 'ruff-on-save)
;;; ruff-on-save.el ends here
