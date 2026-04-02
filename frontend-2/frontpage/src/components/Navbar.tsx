import { motion } from "framer-motion";

const links = ["Home", "Markets", "Analysis", "Login"];

export default function Navbar() {
  return (
    <motion.nav
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
      className="fixed top-0 left-0 right-0 z-50 glass-surface"
    >
      <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
        <span className="font-display text-lg font-semibold tracking-tight text-foreground">
          <span className="text-gradient-brand">AInvestify</span>
        </span>
        <ul className="flex items-center gap-8">
          {links.map((link) => (
            <li key={link}>
              <a
                href="#"
                className="font-display text-sm font-medium text-muted-foreground transition-colors duration-200 hover:text-foreground"
              >
                {link}
              </a>
            </li>
          ))}
        </ul>
      </div>
    </motion.nav>
  );
}
