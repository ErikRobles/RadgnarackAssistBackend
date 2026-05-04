# RadgnarackAssist Embeddable Widget

## Overview

The RadgnarackAssist embeddable widget is a lightweight JavaScript chat widget that can be added to any website with a single script tag.

It connects to the RadgnarackAssist backend and renders a floating AI assistant chat interface on the page.

## Basic Installation

Add this script tag to your website:

```html
<script
  src="https://api.radgnarackassist.rrspark.website/widget/widget.js"
  data-api-base="https://api.radgnarackassist.rrspark.website"
></script>
```

## Configurable Options

The widget supports the following script tag attributes:

- `data-api-base`
- `data-width`
- `data-bottom`
- `data-right`

Default values:

- `data-width="360px"`
- `data-bottom="24px"`
- `data-right="24px"`

### Attribute notes

- `data-api-base` — Base URL of the RadgnarackAssist backend the widget should call.
- `data-width` — Width of the chat panel.
- `data-bottom` — Distance from the bottom edge of the page.
- `data-right` — Distance from the right edge of the page.

## Example With Custom Positioning

```html
<script
  src="https://api.radgnarackassist.rrspark.website/widget/widget.js"
  data-api-base="https://api.radgnarackassist.rrspark.website"
  data-width="420px"
  data-bottom="32px"
  data-right="32px"
></script>
```

## Local Development Test

Run the backend locally:

```bash
./venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

Then load the widget from your local backend using:

```html
<script
  src="http://localhost:8001/widget/widget.js"
  data-api-base="http://localhost:8001"
></script>
```

You can also use the local test page in this repository after starting the backend.

## Backend Endpoint Used

The widget sends requests to:

```text
POST /api/chat
```

Payload example:

```json
{
  "question": "What colors do your racks come in?",
  "conversation_id": "conv_..."
}
```

## Styling Isolation

- No React is required.
- No build step is required.
- Isolated CSS classes use the `ra-widget-` prefix.
- The widget is intended not to interfere with host page styles.

## Notes For Developers

- The widget must be able to reach the backend URL configured in `data-api-base`.
- CORS must allow the host website domain.
- HTTPS is recommended in production.
- By default, the widget appears as a floating button in the bottom-right corner.

## Quick Test Questions

Try these example questions:

- What colors do your racks come in?
- Do they come in blue?
- What hitch do I need?
