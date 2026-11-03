# =====================================================================
#  Makefile — KNN C++ WebAssembly Project
#  Yêu cầu: Emscripten (emcc) đã được cài và activate
# =====================================================================

# ---- Cấu hình -------------------------------------------------------
CXX      = em++
SRCDIR   = src
WEBDIR   = web
BUILDDIR = build
DOCSDIR  = docs

SOURCES  = $(SRCDIR)/knn.cpp $(SRCDIR)/bindings.cpp
OUTPUT   = $(BUILDDIR)/knn.js

# Emscripten flags
EMFLAGS  = -std=c++17 \
           --bind \
           -s MODULARIZE=1 \
           -s EXPORT_NAME="KNNModule" \
           -s ALLOW_MEMORY_GROWTH=1 \
           -s ENVIRONMENT='web' \
           -O2

# =====================================================================
.PHONY: all build serve docs clean help

# ---- Default target -------------------------------------------------
all: build docs

# ---- Build WASM -----------------------------------------------------
build: $(SOURCES)
	@echo "[BUILD] Compiling C++ → WebAssembly..."
	@mkdir -p $(BUILDDIR)
	$(CXX) $(SOURCES) -o $(OUTPUT) $(EMFLAGS)
	@echo "[BUILD] Done: $(OUTPUT) + $(BUILDDIR)/knn.wasm"

# ---- Copy web/ + build/ → docs/ (GitHub Pages) ---------------------
docs: build
	@echo "[DOCS] Copying files to docs/..."
	@mkdir -p $(DOCSDIR)
	@cp $(WEBDIR)/*.html $(WEBDIR)/*.css $(WEBDIR)/*.js $(DOCSDIR)/
	@cp $(BUILDDIR)/knn.js $(BUILDDIR)/knn.wasm $(DOCSDIR)/
	@echo "[DOCS] Done: docs/ is ready for GitHub Pages"

# ---- Serve locally --------------------------------------------------
serve: docs
	@echo "[SERVE] Serving at http://localhost:8080"
	@python -m http.server 8080 --directory $(DOCSDIR)

# ---- Clean ----------------------------------------------------------
clean:
	@echo "[CLEAN] Removing build/ and docs/..."
	@rm -rf $(BUILDDIR) $(DOCSDIR)
	@echo "[CLEAN] Done"

# ---- Help -----------------------------------------------------------
help:
	@echo ""
	@echo "  make build  — Compile C++ → WebAssembly (cần emcc)"
	@echo "  make docs   — Copy files vào docs/ (GitHub Pages)"
	@echo "  make serve  — Chạy local server tại localhost:8080"
	@echo "  make clean  — Xóa build/ và docs/"
	@echo ""
