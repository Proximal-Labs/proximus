;;; check_ruff.el --- Ruff save hook tests -*- lexical-binding: t; -*-
;; Run from the project root: emacs --batch -Q -l scripts/check_ruff.el
(require 'ert)
(require 'python)

(defmacro proximus-with-python-buffer (&rest body)
  `(let* ((enable-local-variables :all)
          (enable-local-eval t) ; Trust only this checkout during the test.
          (make-backup-files nil)
          (auto-save-default nil)
          (file (make-temp-file (expand-file-name "tests/ruff-check-"
                                                  default-directory) nil ".py"))
          (buffer (find-file-noselect file)))
     (unwind-protect
         (with-current-buffer buffer ,@body)
       (with-current-buffer buffer (set-buffer-modified-p nil))
       (kill-buffer buffer)
       (delete-file file))))

(ert-deftest proximus-ruff-save ()
  (proximus-with-python-buffer
   (should (memq #'proximus-ruff-before-save before-save-hook))
   (insert "import sys\nimport os\nprint( sys.version,os.name )\n")
   (save-buffer)
   (should (equal (buffer-string)
                  "import os\nimport sys\n\nprint(sys.version, os.name)\n"))
   (let ((formatted (buffer-string)))
     (proximus-ruff-before-save)
     (should (equal formatted (buffer-string)))
     (should-not (buffer-modified-p))
     (should (equal formatted
                    (with-temp-buffer
                      (insert-file-contents file)
                      (buffer-string)))))))

(ert-deftest proximus-ruff-invalid-input ()
  (proximus-with-python-buffer
   (insert "def broken(\n")
   (should-error (proximus-ruff-before-save))
   (should (equal (buffer-string) "def broken(\n"))))

(ert-deftest proximus-ruff-narrowed-buffer ()
  (proximus-with-python-buffer
   (insert "x=1\ny=2\n")
   (narrow-to-region 5 9)
   (proximus-ruff-before-save)
   (widen)
   (should (equal (buffer-string) "x = 1\ny = 2\n"))))

(ert-deftest proximus-ruff-missing-executable ()
  (proximus-with-python-buffer
   (insert "x=1\n")
   (let ((proximus-ruff-root (make-temp-file "ruff-missing-" t)))
     (unwind-protect
         (progn
           (should-error (proximus-ruff-before-save))
           (should (equal (buffer-string) "x=1\n")))
       (delete-directory proximus-ruff-root)))))

(ert-run-tests-batch-and-exit)
