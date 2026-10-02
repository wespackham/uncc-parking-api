"""Tests for the optional DISCORD_LABEL host prefix."""

from parking_api import config


def test_no_label_leaves_message_unchanged(monkeypatch):
    monkeypatch.setattr(config, "DISCORD_LABEL", "")
    assert config.discord_labeled("hi") == "hi"


def test_label_prefixes_message(monkeypatch):
    monkeypatch.setattr(config, "DISCORD_LABEL", "droplet")
    assert config.discord_labeled("hi") == "**[droplet]** hi"
