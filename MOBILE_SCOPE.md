# Mobile application scope

The delivered interface is responsive at 320 px and desktop widths, with a standalone
web manifest. The original roadmap requested a mobile scope after the web workflow
became dependable; it did not authorize publishing an app.

## Proposed first mobile release

- Reuse the web screens: text/PDF input, extracted-text correction, reviewer evidence,
  graph explanations, accepted edits, PDF/text downloads, and attributed job import.
- Use an explicitly paired private laptop service for Gemma inference. A phone's
  `localhost` is the phone itself; the current loopback-only laptop service is not
  remotely reachable. Pairing, authenticated HTTPS and private networking must precede
  an actual phone-connected release.
- Keep candidate text out of persistent caches. Offline access should cover the
  interface/help only; analysis requires the paired service. No background uploads.
- Clearly show connection state, selected local model, progress, cancellation/retry,
  unavailable research features, and unsupported PDF characters.
- Verify camera-generated scanned PDFs, OS file pickers, downloads/share sheets,
  keyboard navigation, screen readers and narrow-screen editing on real devices.

## Release gates

Private pairing and device testing remain work for a future mobile release. Native
Android/iOS packages, app-store publication, offline on-phone E4B inference and a
service worker are not implemented or advertised. The present deliverable is a local
desktop web application with a responsive interface and manifest.
