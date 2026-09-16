import typography from '@tailwindcss/typography';

/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
  theme: {
    extend: {
      colors: {
        // Natural relaxing light mode & pure black dark mode
        bg: {
          light: '#faf9f6', // natural relaxing paper tone
          dark: '#000000',  // pure black
        },
        card: {
          light: '#ffffff',
          dark: '#0a0a0a',
        },
        border: {
          light: '#e5e5e5',
          dark: '#1e1e1e',
        },
        accent: {
          DEFAULT: '#2563eb', // understated clean blue
          hover: '#1d4ed8',
          amber: '#d97706',
          emerald: '#059669',
          rose: '#e11d48',
          violet: '#6d28d9',
        }
      },
      fontFamily: {
        sans: ['Inter', 'Geist', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['"JetBrains Mono"', '"Fira Code"', 'monospace'],
      },
      borderRadius: {
        none: '0px',
        sm: '2px',
        DEFAULT: '4px',
        md: '6px',
        lg: '8px',
        xl: '12px',
      },
      typography: (theme) => ({
        DEFAULT: {
          css: {
            maxWidth: '100%',
            color: 'inherit',
            a: {
              color: theme('colors.blue.600'),
              textDecoration: 'underline',
              textUnderlineOffset: '3px',
              '&:hover': {
                color: theme('colors.blue.700'),
              },
            },
            'h1, h2, h3, h4': {
              letterSpacing: '-0.02em',
              fontWeight: '700',
            },
            code: {
              color: 'inherit',
              backgroundColor: 'rgba(0, 0, 0, 0.05)',
              paddingLeft: '0.3rem',
              paddingRight: '0.3rem',
              paddingTop: '0.1rem',
              paddingBottom: '0.1rem',
              borderRadius: '3px',
              fontWeight: '400',
            },
            '.dark code': {
              backgroundColor: 'rgba(255, 255, 255, 0.1)',
            },
            'code::before': {
              content: '""',
            },
            'code::after': {
              content: '""',
            },
          },
        },
      }),
    },
  },
  plugins: [typography],
};
