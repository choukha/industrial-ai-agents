"""Entry point for the IDOCA application."""

import argparse
import logging
import sys

from idoca.interface import create_interface
from idoca.config import LOG_FORMAT

def setup_logging(level=logging.INFO):
    """Configure logging for the application."""
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("idoca.log", mode="a")
        ]
    )
    return logging.getLogger("idoca")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Industrial Documents Analysis Agent")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to serve on")
    parser.add_argument("--port", type=int, default=7860, help="Port to serve on")
    parser.add_argument("--share", action="store_true", help="Create a public URL")
    return parser.parse_args()

def main():
    """Run the IDOCA application."""
    args = parse_args()
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(level=log_level)
    
    logger.info("Starting Industrial Documents Analysis Agent (IDOCA)")
    logger.info(f"Debug mode: {'Enabled' if args.debug else 'Disabled'}")
    
    try:
        # Create and launch the Gradio interface
        interface = create_interface()
        interface.queue().launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            debug=args.debug
        )
    except Exception as e:
        logger.error(f"Error starting application: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()