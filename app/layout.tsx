import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Folio · Your portfolio, clearly",
  description: "A private, free-first portfolio dashboard for stocks and mutual funds.",
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
