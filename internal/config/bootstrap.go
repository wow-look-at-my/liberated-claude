package config

import (
	"encoding/xml"
	"fmt"
	"strings"
)

// Bootstrap is the overlay Claude Desktop reads. Element names are the app's
// own setting keys, passed through untranslated.
type Bootstrap struct {
	Settings []Setting `xml:",any"`
}

// Setting is one overlay key. An element with child elements carries them in
// Children and ignores Value; a leaf carries its text in Value.
type Setting struct {
	XMLName  xml.Name
	Value    string    `xml:",chardata"`
	Children []Setting `xml:",any"`
}

// Name is the element name, which is the app's setting key.
func (s Setting) Name() string { return s.XMLName.Local }

// JSON renders the overlay as the app's document.
func (b Bootstrap) JSON() map[string]any {
	return settingsJSON(b.Settings)
}

func settingsJSON(settings []Setting) map[string]any {
	out := make(map[string]any, len(settings))
	for _, s := range settings {
		out[s.Name()] = s.value()
	}
	return out
}

// value renders one setting: an array when every child is <item>, a nested
// object when it has other children, a scalar otherwise.
func (s Setting) value() any {
	if len(s.Children) == 0 {
		return scalar(strings.TrimSpace(s.Value))
	}
	if !s.isList() {
		return settingsJSON(s.Children)
	}
	out := make([]any, 0, len(s.Children))
	for _, c := range s.Children {
		out = append(out, c.value())
	}
	return out
}

// isList reports whether the children spell a list rather than an object.
func (s Setting) isList() bool {
	for _, c := range s.Children {
		if c.Name() != "item" {
			return false
		}
	}
	return true
}

func scalar(v string) any {
	switch v {
	case "true":
		return true
	case "false":
		return false
	}
	return v
}

// Find returns the named top-level setting.
func (b Bootstrap) Find(name string) (Setting, bool) {
	for _, s := range b.Settings {
		if s.Name() == name {
			return s, true
		}
	}
	return Setting{}, false
}

// Bool reports a top-level boolean setting, and whether it was supplied at all.
func (b Bootstrap) Bool(name string) (value, ok bool) {
	s, found := b.Find(name)
	if !found {
		return false, false
	}
	return strings.TrimSpace(s.Value) == "true", true
}

// child returns a nested element's text.
func (s Setting) child(name string) string {
	for _, c := range s.Children {
		if c.Name() == name {
			return strings.TrimSpace(c.Value)
		}
	}
	return ""
}

// validateImport enforces the two rules Claude Desktop applies to
// claudeAiImport, so a rejected overlay surfaces here by name.
//
// The endpoint trio is an override for a self-hosted export service and is set
// together or left entirely empty. Empty with enabled=true is the normal case:
// the app imports from Claude.ai using its own endpoints.
func (b Bootstrap) validateImport() error {
	imp, ok := b.Find("claudeAiImport")
	if !ok {
		return nil
	}

	switch banner := imp.child("bannerBehavior"); banner {
	case "", "off", "detect", "show":
	default:
		return fmt.Errorf(
			"bootstrap.claudeAiImport.bannerBehavior %q is not one of off, detect, show",
			banner)
	}

	endpoint := []string{"url", "oauthIssuer", "oauthClientId"}
	var set, missing []string
	for _, name := range endpoint {
		if imp.child(name) != "" {
			set = append(set, name)
			continue
		}
		missing = append(missing, name)
	}
	if len(set) > 0 && len(missing) > 0 {
		return fmt.Errorf(
			"bootstrap.claudeAiImport: %s set without %s (the export URL, OAuth issuer and client ID are set together or left entirely empty)",
			strings.Join(set, ", "), strings.Join(missing, ", "))
	}
	return nil
}
