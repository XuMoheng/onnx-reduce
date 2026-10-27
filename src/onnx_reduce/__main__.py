"""Allow ``python -m onnx_reduce`` in addition to the installed script."""

from onnx_reduce.cli import main

raise SystemExit(main())
