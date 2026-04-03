#!/bin/bash

echo ""
echo "Paste your OpenAQ API key and press ENTER:"
read KEY

echo "OPENAQ_API_KEY=$KEY" > .env

echo ""
echo "Saved API key to .env"
