import { Separator } from "@opencut/components/ui/separator";
import { type Tab, useAssetsPanelStore } from "@opencut/components/editor/panels/assets/assets-panel-store";
import { TabBar } from "./tabbar";
// Recall: captions come from Recall's own transcript, not in-browser Whisper.
import { RecallCaptionsView } from "@opencut/recall/captions-view";
import { RecallMomentsView } from "@opencut/recall/moments-view";
import { useRecallHost } from "@opencut/recall/host";
import { useEffect } from "react";
import { MediaView } from "./views/assets";
import { SettingsView } from "./views/settings";
import { SoundsView } from "@opencut/sounds/components/assets-view";
import { StickersView } from "@opencut/stickers/components/assets-view";
import { TextView } from "@opencut/text/components/assets-view";
import { EffectsView } from "@opencut/effects/components/assets-view";

export function AssetsPanel() {
	const { activeTab, setActiveTab } = useAssetsPanelStore();
	// Recall: the Cutting Room opens on its Moments; elsewhere that tab doesn't exist.
	const recallHost = useRecallHost();
	const hasMoments = !!recallHost.moments;
	const hasCaptions = !!recallHost.captions;
	useEffect(() => {
		if (hasMoments) setActiveTab("moments");
		else if (useAssetsPanelStore.getState().activeTab === "moments") setActiveTab("media");
		// A tab this editor doesn't show (persisted from before) falls back to Media.
		const shown = useAssetsPanelStore.getState().activeTab;
		if ((shown === "captions" && !hasCaptions) || shown === "transitions" || shown === "adjustment") setActiveTab(hasMoments ? "moments" : "media");
	}, [hasMoments, hasCaptions, setActiveTab]);

	const viewMap: Record<Tab, React.ReactNode> = {
		moments: <RecallMomentsView />,
		media: <MediaView />,
		sounds: <SoundsView />,
		text: <TextView />,
		stickers: <StickersView />,
		effects: <EffectsView />,
		transitions: (
			<div className="text-muted-foreground p-4">
				Transitions view coming soon...
			</div>
		),
		captions: <RecallCaptionsView />,
		adjustment: (
			<div className="text-muted-foreground p-4">
				Adjustment view coming soon...
			</div>
		),
		settings: <SettingsView />,
	};

	return (
		<div className="panel bg-background flex h-full rounded-sm border overflow-hidden">
			<TabBar />
			<Separator orientation="vertical" />
			<div className="flex-1 overflow-hidden">{viewMap[activeTab]}</div>
		</div>
	);
}
