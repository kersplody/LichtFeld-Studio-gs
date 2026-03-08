/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/panel_layout.hpp"
#include "gui/rmlui/rml_fbo.hpp"
#include <RmlUi/Core/DataModelHandle.h>
#include <cstddef>
#include <string>
#include <vector>

namespace Rml {
    class Context;
    class ElementDocument;
    class Element;
} // namespace Rml

namespace lfs::vis {
    struct Theme;
}
namespace lfs::vis::gui {

    class RmlUIManager;

    struct MenuItemDesc {
        enum class Type { Operator,
                          Separator,
                          SubMenuBegin,
                          SubMenuEnd,
                          Toggle,
                          ShortcutItem,
                          Item };
        Type type;
        std::string label;
        std::string operator_id;
        std::string shortcut;
        bool enabled = true;
        bool selected = false;
        int callback_index = -1;
    };

    struct MenuDropdownContent {
        std::string menu_idname;
        std::vector<MenuItemDesc> items;
    };

    struct MenuLabelView {
        int index = -1;
        std::string label;
        bool active = false;
    };

    struct MenuDropdownLeafView {
        std::string label;
        std::string action;
        std::string operator_id;
        std::string shortcut;
        std::string checkmark;
        bool enabled = true;
        bool separator_before = false;
        bool has_shortcut = false;
        bool show_checkmark = false;
        bool is_label = false;
        int callback_index = -1;
    };

    struct MenuDropdownRootView {
        std::string label;
        std::string action;
        std::string operator_id;
        std::string shortcut;
        std::string checkmark;
        bool enabled = true;
        bool separator_before = false;
        bool has_shortcut = false;
        bool show_checkmark = false;
        bool has_children = false;
        int callback_index = -1;
        std::vector<MenuDropdownLeafView> children;
    };

    class RmlMenuBar {
    public:
        void init(RmlUIManager* mgr);
        void shutdown();
        void draw(int screen_w, int screen_h);
        void updateLabels(const std::vector<std::string>& labels,
                          const std::vector<std::string>& idnames);
        void processInput(const PanelInputState& input);
        bool wantsInput() const { return wants_input_; }
        bool isOpen() const { return open_menu_index_ >= 0; }
        const RmlFBO& fbo() const { return fbo_; }
        float barHeight() const;

    private:
        void updateTheme();
        void rebuildLabels();
        void syncActiveLabelState();
        std::string generateThemeRCSS(const lfs::vis::Theme& t) const;
        void openDropdown(int index);
        void closeDropdown();
        void rebuildDropdownDOM();

        RmlUIManager* rml_manager_ = nullptr;
        Rml::Context* rml_context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;
        Rml::DataModelHandle menu_model_;

        RmlFBO fbo_;

        std::size_t last_theme_signature_ = 0;
        bool has_theme_signature_ = false;
        std::string base_rcss_;

        std::vector<std::string> current_labels_;
        std::vector<std::string> current_idnames_;
        std::vector<MenuLabelView> menu_labels_;
        std::vector<MenuDropdownRootView> dropdown_items_;
        int active_index_ = -1;

        Rml::Element* menu_items_ = nullptr;
        Rml::Element* dropdown_container_ = nullptr;
        Rml::Element* dropdown_overlay_ = nullptr;

        int open_menu_index_ = -1;
        std::string open_menu_idname_;
        bool wants_input_ = false;

        float bar_height_ = 30.0f;
    };

} // namespace lfs::vis::gui
