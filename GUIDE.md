# Role & Goal
You are an expert full-stack developer and world-class UI/UX designer specializing in high-performance, aesthetically pleasing developer blogs and portfolios. You will build a technical blog and personal site using Astro (SSG), Tailwind CSS, Shiki, KaTeX, and Lucide Icons. The site will be hosted on Cloudflare Pages.

# Target Vibe & Design Principles
- Inspired by Andrej Karpathy (karpathy.ai) and Arpit Bhayani: ultra-clean, authoritative, reading-focused, and highly polished.
- Impeccable dark/light mode execution: Soft, non-glare dark mode (slate/zinc tones, not pitch black `#000000`) and a warm, low-contrast light mode.
- Fast performance, zero-layout-shift (CLS), and 100/100 Core Web Vitals.

---

# Required Pages & Structural Breakdown

## Page 1: Main Home / Portfolio Page (`/`)
1. **Hero Header:**
   - High-quality avatar/profile photo of me with clean rounded borders or a subtle accent ring.
   - Bio: Name, current role, domain focus (LLM inference runtime engineering, systems optimizations, hardware/compilers).
   - Social Icons Bar: Integrated links to X (Twitter), GitHub, Substack, and Email using crisp icons.
2. **"My Journey & About Me" Section:**
   - A short, compelling background narrative detailing my transition, engineering focus, and current technical pursuits.
3. **Featured / Flagship Writings:**
   - A curated grid/list of top-tier blog posts with reading time, date, tags, and summary.
4. **Publications & Research Section:**
   - Dedicated section highlighting paper publications, technical reports, open-source documentation, or major specs.
   - Includes title, venue/platform, publication date, abstract snippet, and external link (PDF, arXiv, or repo).
5. **Topics / Tags Overview:**
   - Quick pills/chips to filter writing by topic (e.g., `#systems`, `#inference`, `#compilers`, `#cuda`).

## Page 2: Individual Blog Post Page (`/posts/[slug]`)
1. **Reading Experience & Aesthetics:**
   - Optimized typography using a dual-font strategy: Inter/Geist for prose and JetBrains Mono/Fira Code for code blocks.
   - Optimal line length (`max-w-3xl`) to maximize reading comfort and retention.
   - Embedded support for rich media: image captions, inline architectural diagrams, and video embeds.
2. **Interactive Elements:**
   - Floating or sticky Table of Contents (ToC) on desktop so readers can jump between sections seamlessly.
   - Code Blocks: Syntax highlighting powered by Shiki, explicit code language badge, line numbers, line highlighting, and a one-click "Copy Code" button.
   - Callout Component Cards for `💡 Note`, `⚡ Optimization Tip`, and `⚠️ Bottleneck Warning`.
   - Native KaTeX rendering for math equations.
3. **Cross-Linking & Related Content:**
   - "Related Articles" section at the end of every post featuring 2–3 relevant reads based on post tags.
   - Previous / Next article navigation links.
   - Embedded social sharing & email newsletter subscription box at the bottom of each post.

## Page 3: Navigation & Global Layout (`Header` & `Footer`)
1. **Simplistic Header Nav:**
   - Minimalist sticky navigation bar with blurred background (`backdrop-blur`).
   - Clean links: `Home`, `Blog`, `Publications`, `About`.
   - Theme toggle (Light/Dark mode) with smooth icon transition.
2. **Comprehensive Footer:**
   - Interlinked site sitemap divided into logical columns: *Navigation*, *Categories*, *Socials & Contact*.
   - Dynamic RSS Feed link (`/rss.xml`).
   - Minimal copyright notice.

---

# Strategic Social & Link Integration
- Social links placed in three intentional locations:
  1. Main page Hero header (Primary profile links).
  2. Individual blog post footer (For readers who just finished a deep dive).
  3. Global site footer.

---

# Technical Implementation Guidelines (Astro + Tailwind)

1. **Astro Setup:**
   - Use Content Collections (`src/content/config.ts`) for both `posts` and `publications` schemas.
   - Content Collection Schemas:
     - **Posts:** `title`, `description`, `pubDate`, `updatedDate`, `heroImage`, `tags`, `featured` (boolean).
     - **Publications:** `title`, `venue`, `pubDate`, `abstract`, `link`, `paperUrl`.

2. **Tailwind Typography Configuration:**
   - Add `@tailwindcss/typography` plugin.
   - Customize dark/light mode contrast in `tailwind.config.mjs` for seamless code block readability.

3. **SEO & Meta Tags Component (`SEO.astro`):**
   - Dynamic `<title>`, `<meta name="description">`, OpenGraph images, Twitter Card tags, and canonical links.
   - Automatically generate `sitemap.xml` and `rss.xml` on build.

4. **Sample Data:**
   - Include 2 sample Markdown posts with C++ code snippets, benchmark tables, callouts, and images to test the UI layout.
   - Include 1 sample publication entry.