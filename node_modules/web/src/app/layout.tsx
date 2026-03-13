import type { Metadata } from "next";
import { Bricolage_Grotesque, Cormorant_Garamond } from "next/font/google";
import NextTopLoader from "nextjs-toploader";

import "./globals.css";

const bodyFont = Bricolage_Grotesque({
  variable: "--font-body",
  subsets: ["latin"],
});

const displayFont = Cormorant_Garamond({
  variable: "--font-display",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

export const metadata: Metadata = {
  title: "GenZ AI Therapist",
  description:
    "A Gen Z-coded, non-clinical AI listener for yapping, journaling, vibe checks, and helpful support resources.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${bodyFont.variable} ${displayFont.variable}`}>
        <NextTopLoader
          color="#35584e"
          crawl={true}
          crawlSpeed={180}
          height={3}
          initialPosition={0.18}
          showSpinner={false}
          easing="ease"
          speed={220}
          shadow="0 0 10px rgba(53, 88, 78, 0.35)"
        />
        {children}
      </body>
    </html>
  );
}
