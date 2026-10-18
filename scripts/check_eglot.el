;;; check_eglot.el --- Project-local Eglot smoke test -*- lexical-binding: t; -*-
;; Run from the project root: emacs --batch -Q -l scripts/check_eglot.el
(require 'ert)
(require 'eglot)
(require 'python)

(ert-deftest proximus-eglot ()
  (let* ((enable-local-variables :all) ; Trust this checkout's .dir-locals.el.
         (enable-local-eval nil)
         (eglot-sync-connect t)
         (eglot-connect-timeout 30)
         (eglot-autoshutdown nil)
         (file (make-temp-file (expand-file-name "eglot-check-" default-directory)
                               nil ".py"))
         buffer server)
    (unwind-protect
        (progn
          (with-temp-file file
            (insert "from pydantic import BaseModel\n"
                    "from harbor_proximus.timer import write_deadline\n"
                    "bad: int = 'wrong'\n"
                    "write_deadline\n"))
          (setq buffer (find-file-noselect file))
          (with-current-buffer buffer
            (should (equal (cdr (assoc '(python-mode python-ts-mode)
                                      eglot-server-programs))
                           '("uv" "run" "--locked" "pyright-langserver" "--stdio")))
            (call-interactively #'eglot)
            (setq server (eglot-current-server))
            (should server)
            (flymake-start)
            (goto-char (point-min))
            (forward-line 3)
            (forward-char 5)
            (let* ((params (eglot--TextDocumentPositionParams))
                   (hover (jsonrpc-request server :textDocument/hover params))
                   (definitions (jsonrpc-request server :textDocument/definition params))
                   (completion (jsonrpc-request server :textDocument/completion params)))
              (should (string-match-p "write_deadline" (prin1-to-string hover)))
              (should (string-match-p "harbor_proximus/timer.py"
                                      (prin1-to-string definitions)))
              (should (string-match-p "write_deadline" (prin1-to-string completion))))
            (let ((deadline (+ (float-time) 30)))
              (while (and (null (flymake-diagnostics)) (< (float-time) deadline))
                (accept-process-output nil 0.1)))
            (let ((messages (mapcar #'flymake-diagnostic-text (flymake-diagnostics))))
              (should (seq-some (lambda (message) (string-match-p "int" message))
                                messages))
              (should-not (seq-some (lambda (message)
                                     (string-match-p "could not be resolved" message))
                                   messages)))))
      (when server
        (with-current-buffer buffer (eglot-shutdown server)))
      (when (buffer-live-p buffer) (kill-buffer buffer))
      (delete-file file))))

(ert-run-tests-batch-and-exit)
